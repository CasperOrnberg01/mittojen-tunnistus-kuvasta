# app/services/a4_detection.py
# A4 paper sheet detection logic
# Improved version: new import (itertools) + func. detect_a4() sligthly modified + new helper functions
# Current known flaws/bugs: papermask and visible corner fallback methods might give false A4 detections when no A4 in image->
# Fixes have been tried, but they can affect some good A4 shots as well so for now, keeping logic the same

# New import, to generate combinations of four detected corner points
import itertools

import cv2 # OpenCV
import numpy as np #Numpy lib

# A4 aspect ratio reference 297mm / 210mm = 1.414...
A4_RATIO = 297 / 210


def order_corners(pts):
    """
    Put four corner points into a stable order:
    [top-left, top-right, bottom-right, bottom-left].

    This order is required for perspective correction.
    If the points are in a random order, cv2.getPerspectiveTransform()
    may twist or flip the warped A4 image.
    """

    # Reshape the input into exactly four (x, y) coordinate pairs
    # astype("float32") is required by OpenCV perspective transform functions
    pts = pts.reshape(4, 2).astype("float32")

    #Find center of 4 points, sort points around that center by angle
    # -> rotate ordered list so first point is the image space TL corner
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # Start from the most top left like point: tl, tr, br, bl 
    start_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start_index, axis=0)

    # Make sure the points are in tl, tr, br, bl order
    # In image coordinates, y grows downward. For the expected order, the signed
    # polygon area should be positive. If it is negative --> reverse direction
    # while keeping same first corner (tl)
    signed_area = 0.0
    for i in range(4):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % 4]
        signed_area += (x1 * y2) - (x2 * y1)

    if signed_area < 0:
        ordered = np.array([
            ordered[0],
            ordered[3],
            ordered[2],
            ordered[1],
        ], dtype="float32")

    # Return the ordered corners as float32 for later OpenCV operations
    return ordered.astype("float32")


def detect_a4(image):
    """
    Detect an A4 sheet in an OpenCV image.

    Returns:
      {"a4_found": True, ...} when a likely A4 candidate is found.
      {"a4_found": False} when no reliable candidate is found.
    """

    # Convert image to grayscale = reduces complexity in detection hence color is not useful for shape detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast with CLAHE
    # clipLimit=2.0 limits how aggressively contrast is boosted, 2.0 safe value to start with
    # tileGridSize=(8, 8) splits the image into small regions for local contrast 
    # 8x8 is commonly used because it balances local detail and smoothness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur image for noise reduction
    # GaussianBlur smooth image and removes small noise
    # kernel size set to (5,5)   must be odd numbers for the best results
    # standard deviation 0 = Lets OpenCV automatically calculate the best standard deviation based on the kernel size
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges with the Canny algorithm. 50 lower treshold, 150 upper treshold
    # Edges stronger than 150 are kept, edges between 50 and 150 are kept if connected to stronger edges
    edges = cv2.Canny(blurred, 50, 150)

    # Create a rectangular morphology kernel
    # (5, 5) means small gaps up to a few pixels can be closed
    # This helps turn broken paper edge segments into more complete contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Close small gaps in the detected edges
    # MORPH_CLOSE performs dilation followed by erosion
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours from the edge image
    # cv2.RETR_LIST returns both outer and inner contours
    # cv2.CHAIN_APPROX_SIMPLE stores only important contour points to save memory
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST, # note: fetch also inner contours, not only outers. Before I used cv2.RETR_EXTERNAL that caused false positives by fetching outer contour = the whole image
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours by area from largest to smallest
    # Large rectangles are more likely to be paper than tiny texture/noise contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Track the best candidate found so far. A score of 0 = no candidate has been accepted yet
    best_score = 0

    # Stores the metadata of the best A4 candidate
    best_candidate = None

    # Stores the contour and rotated rectangle for the best candidate
    # The contour is used first for corner estimation, then minAreaRect is used as a fallback
    best_contour = None
    best_rect = None

    # image.shape is (height, width, channels) for a color image
    # We only need height and width here
    h, w = image.shape[:2]

    # Total image area in pixels
    # Used to compare contour size relative to the whole photo
    image_area = h * w

    # Examine each contour and decide whether it could be the A4 paper
    for contour in contours:
        # Calculate contour area in pixels
        area = cv2.contourArea(contour)

        # Ignore very small contours
        # 0.01 = 1% of the whole image, anything smaller unlikely to be full A4 sheet and is often texure or noise
        if area < image_area * 0.01:
            continue

        # Calculate how much of the whole image this contour covers.
        area_ratio = area / image_area

        # Reject contours that are almost the whole image, 0.85 = 85% of image
        # This prevents the detector from selecting the photo border or carpet boundary as A4
        if area_ratio > 0.85:
            continue

        # Fit the smallest rotated rectangle around the contour
        # This is useful because A4 may be rotated in the photo
        rect = cv2.minAreaRect(contour)

        # minAreaRect returns:
        # (x, y): center point of the rectangle
        # (rw, rh): rectangle width and height
        # angle: rotation angle of the rectangle
        (x, y), (rw, rh), angle = rect

        # If width or height is zero, the rectangle is invalid
        # This avoids division by zero when calculating aspect ratio
        if rw == 0 or rh == 0:
            continue

        # Calculate shape ratio using longer side / shorter side
        # This makes portrait and landscape A4 both produce about 1.414
        ratio = max(rw, rh) / min(rw, rh)

        # Reject shapes too far from the A4 ratio
        # 0.25 allows some error from perspective, imperfect contours, and camera angle
        # Target is about 1.414, so accepted range is roughly 1.164 to 1.664
        if abs(ratio - A4_RATIO) > 0.25:
            continue

        # Create the convex hull around the contour. Hull is the smallest convex shape that contains the contour
        hull = cv2.convexHull(contour)

        # Calculate hull area in pixels
        hull_area = cv2.contourArea(hull)

        # Avoid division by zero
        if hull_area == 0:
            continue

        # Solidity measures how much the contour fills its convex hull
        # A clean paper rectangle should be solid and close to 1.0
        solidity = area / hull_area

        # Reject contours that are too irregular or broken. 0.92 = 92% meaning contour must fill at least 92% of its convex hull
        # This keeps the detector focused on rectangle like paper shapes
        if solidity < 0.92:
            continue

        # Create a black mask with the same size as the grayscale image
        # The mask will mark only the area inside the current contour
        mask = np.zeros_like(gray)

        # Fill the current contour with white on the mask
        # -1 means draw all contour points
        # 255 means white in an 8-bit image
        # -1 as thickness means fill the contour
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Count edge pixels inside the contour
        # bitwise_and keeps only edge pixels where the mask is white
        edge_inside = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))

        # Count all pixels inside the contour mask
        contour_pixels = cv2.countNonZero(mask)

        # Avoid division by zero
        if contour_pixels == 0:
            continue

        # Edge density means how much edge/noise exists inside the candidate area
        # A plain A4 sheet should not have extreme internal edge density
        edge_density = edge_inside / contour_pixels

        # Reject candidates that are too edge heavy inside
        # 0.85 is very permissive, but helps remove noisy regions that are mostly texture
        if edge_density > 0.85:
            continue

        # Score the candidate
        # Larger area is better, whereas higher solidity is better too. Smaller aspect ratio error is better.
        # +0.01 prevents division by zero if ratio matches A4 almost exactly
        score = area * solidity / (abs(ratio - A4_RATIO) + 0.01)

        # Keep this contour if it is better than the previous best candidate
        if score > best_score:
            best_score = score
            best_contour = contour
            best_rect = rect
            best_candidate = {
                "ratio": ratio,
                "angle": angle,
                "score": score,
                "solidity": solidity,
                "area": area,
                "image_area": image_area,
                "area_ratio": area_ratio,
            }

    # If a valid candidate was found, convert it into API friendly output
    if best_candidate and best_rect is not None:
        # Try to estimate real contour corners first
        # If that is not reliable, fall back to minAreaRect corners
        ordered = _corners_from_contour_or_rect(best_contour, best_rect, image_area)

        # Return all values needed by debug overlays, warp logic, and future measurement code
        return {
            "a4_found": True,
            "approx_ratio": float(best_candidate["ratio"]),
            "confidence_score": float(best_candidate["score"]),
            "angle": float(best_candidate["angle"]),
            "corners_px": ordered.tolist(),
            "area": float(best_candidate["area"]),
            "image_area": float(best_candidate["image_area"]),
            "area_ratio": float(best_candidate["area_ratio"]),
            "detection_method": "contour",
        }

    # If the clean contour detector fails, try matching the four visible
    # paper corners before using the color based papermask fallback
    # This is aimed at the difficult hand on paper cases where:
    # - all four A4 corners are still visible
    # - the hand interrupts the paper contour
    # - bright reflections or background areas pollute the paper mask
    corner_result = _detect_a4_by_visible_corner_fallback(image)

    if corner_result.get("a4_found", False):
        return corner_result

    # If visible corner matching also fails, keep the original papermask
    # fallback as the final automatic backup
    fallback_result = _detect_a4_by_paper_mask_fallback(image)

    if fallback_result.get("a4_found", False):
        return fallback_result

    # No candidate passed all filters
    return {
        "a4_found": False,
        "needs_manual_corners": True,
        "reason": "No reliable A4 candidate found"
    }


def _detect_a4_by_visible_corner_fallback(image):
    """
    New det. method: Detect A4 from four visible image corners
    Original papermask fallback can work perfectly but connected reflection, light backgr.
    or lighter arm can pull the convex hull and approxPolyDP corners away from the real sheet
    

    This fallback uses a different source of evidence:
      1. Detect strong image corners with goodFeaturesToTrack
      2. Build plausible 4 corner combinations
      3. Keep only A4 like convex geometry
      4. Require paper like pixels mostly inside the shape
      5. Require a paper2background change across several proposed sides
      6. Reject highly textured interiors such as carpet or background rectangles

    runs only after the original clean contour detector fails
    """

    original_h, original_w = image.shape[:2]
    original_max_dim = max(original_h, original_w)

    # Work on a smaller image so corner combinations stay practical
    # Returned corner coordinates are scaled back to the original image
    max_work_dim = 700
    scale = min(1.0, max_work_dim / float(original_max_dim))

    if scale < 1.0:
        work_image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        work_image = image.copy()

    h, w = work_image.shape[:2]
    min_dim = min(h, w)
    image_area = h * w

    gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ask OpenCV for more corners than we finally use
    # goodFeaturesToTrack can return a slightly different ordered subset when
    # maxCorners changes, so detecting 40 and then keeping 26 was more stable
    # than directly asking for exactly 28
    detected_corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=40,
        qualityLevel=0.006,
        minDistance=max(7, int(min_dim * 0.022)),
        blockSize=7,
        useHarrisDetector=False
    )

    if detected_corners is None:
        return {"a4_found": False}

    corner_points = detected_corners.reshape(-1, 2).astype("float32")[:26]

    if len(corner_points) < 4:
        return {"a4_found": False}

    raw_mask = _create_adaptive_paper_mask(work_image)
    cleaned_mask = _clean_paper_mask(raw_mask, work_image.shape[:2])
    paper_likelihood = _create_continuous_paper_likelihood(work_image)

    # Precompute texture maps once to avoid slow processing time
    texture_gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
    texture_blurred = cv2.GaussianBlur(texture_gray, (5, 5), 0)
    texture_gradient_x = cv2.Sobel(texture_blurred, cv2.CV_32F, 1, 0, ksize=3)
    texture_gradient_y = cv2.Sobel(texture_blurred, cv2.CV_32F, 0, 1, ksize=3)
    texture_gradient_magnitude = cv2.magnitude(
        texture_gradient_x,
        texture_gradient_y
    )
    texture_edges = cv2.Canny(texture_blurred, 50, 150)

    # First perform only cheap geometry checks on all combinations
    # 26 choose 4 is about 15,000 combinations, but most are rejected before
    # any mask drawing or side sampling is performed
    geometry_candidates = []

    for indexes in itertools.combinations(range(len(corner_points)), 4):
        corners = order_corners(corner_points[list(indexes)])

        geometry = _visible_corner_geometry_score(
            corners=corners,
            image_shape=work_image.shape[:2]
        )

        if geometry is None:
            continue

        geometry_candidates.append({
            "corners": corners,
            "geometry": geometry,
        })

    if not geometry_candidates:
        return {"a4_found": False}

    # Evaluate only the strongest geometry candidates with the more
    # expensive image evidence checks
    geometry_candidates = sorted(
        geometry_candidates,
        key=lambda item: item["geometry"]["score"],
        reverse=True
    )[:120]

    best_result = None
    best_score = 0.0

    for candidate in geometry_candidates:
        scored = _score_visible_corner_quad(
            work_image=work_image,
            raw_mask=raw_mask,
            cleaned_mask=cleaned_mask,
            paper_likelihood=paper_likelihood,
            texture_gradient_magnitude=texture_gradient_magnitude,
            texture_edges=texture_edges,
            corners=candidate["corners"],
            geometry=candidate["geometry"],
            image_area=image_area
        )

        if scored is None:
            continue

        if scored["score"] > best_score:
            best_score = scored["score"]
            best_result = scored

    if best_result is None:
        return {"a4_found": False}

    # This fallback is intentionally conservative
    # The threshold and texture checks prevent random carpet, table corners from
    # becoming a rectangle only because their geometry happens to resemble A4
    if best_result["score"] < 50.0:
        return {"a4_found": False}

    final_corners = best_result["corners"] / scale
    final_corners = order_corners(final_corners)

    return {
        "a4_found": True,
        "approx_ratio": float(best_result["ratio"]),
        "confidence_score": float(best_result["score"]),
        "angle": float(best_result["angle"]),
        "corners_px": final_corners.tolist(),
        "area": float(best_result["area"] / (scale * scale)),
        "image_area": float(original_h * original_w),
        "area_ratio": float(best_result["area_ratio"]),
        "visible_paper_ratio": float(best_result["paper_fill"]),
        "corner_support": float(best_result["corner_support"]),
        "side_support": float(best_result["side_support"]),
        "boundary_contrast": float(best_result["boundary_contrast"]),
        "supported_boundary_sides": int(best_result["supported_boundary_sides"]),
        "flat_interior_ratio": float(best_result["flat_interior_ratio"]),
        "detection_method": "visible_corner_fallback",
    }


def _visible_corner_geometry_score(corners, image_shape):
    """New fast geometry filter used before consuming mask andcontrast scoring"""

    h, w = image_shape
    image_area = h * w

    if not _is_valid_quad(corners, image_shape):
        return None

    area = _quad_area(corners)
    area_ratio = area / image_area

    # The fallback is for a clearly photographed A4 reference sheet
    # Reject very small corner combinations around fingers or background detail
    if area_ratio < 0.12 or area_ratio > 0.84:
        return None

    tl, tr, br, bl = corners.astype("float32")

    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))

    if min(top, right, bottom, left) < max(12.0, min(h, w) * 0.08):
        return None

    ratio = _quad_side_ratio(corners)
    ratio_error = abs(ratio - A4_RATIO)

    if ratio_error > 0.38:
        return None

    horizontal_balance = min(top, bottom) / max(top, bottom)
    vertical_balance = min(left, right) / max(left, right)

    # Allow perspective, but reject triangles with one collapsed side
    if horizontal_balance < 0.45 or vertical_balance < 0.45:
        return None

    angles = [
        _corner_angle_degrees(bl, tl, tr),
        _corner_angle_degrees(tl, tr, br),
        _corner_angle_degrees(tr, br, bl),
        _corner_angle_degrees(br, bl, tl),
    ]

    if min(angles) < 45.0 or max(angles) > 135.0:
        return None

    average_angle_error = float(np.mean([abs(angle - 90.0) for angle in angles]))

    # This score is only for ranking geometry before image evidence is checked
    score = (
        (3.0 / (ratio_error + 0.08)) +
        (2.0 * (horizontal_balance + vertical_balance)) -
        (0.03 * average_angle_error) +
        (0.50 * min(area_ratio / 0.35, 1.0))
    )

    return {
        "score": float(score),
        "area": float(area),
        "area_ratio": float(area_ratio),
        "ratio": float(ratio),
        "ratio_error": float(ratio_error),
        "horizontal_balance": float(horizontal_balance),
        "vertical_balance": float(vertical_balance),
    }


def _score_visible_corner_quad(
    work_image,
    raw_mask,
    cleaned_mask,
    paper_likelihood,
    texture_gradient_magnitude,
    texture_edges,
    corners,
    geometry,
    image_area
):
    """
    New function to score a four corner candidate using independent evidence

    A valid result needs:
      - enough paper resembling area inside
      - a reasonably smooth interior, even with a hand on the paper
    """

    h, w = work_image.shape[:2]

    quad_mask = np.zeros((h, w), dtype="uint8")
    cv2.fillConvexPoly(quad_mask, corners.astype("int32"), 255)

    quad_pixels = cv2.countNonZero(quad_mask)

    if quad_pixels == 0:
        return None

    paper_inside = cv2.countNonZero(
        cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=quad_mask)
    )
    paper_fill = paper_inside / quad_pixels

    if paper_fill < 0.25:
        return None

    corner_support, _ = _corner_support_score(raw_mask, corners)
    side_support, _ = _side_support_score(raw_mask, corners)

    boundary_scores = _paper_boundary_contrast_scores(
        paper_likelihood=paper_likelihood,
        corners=corners
    )

    sorted_boundary_scores = sorted(boundary_scores, reverse=True)

    strong_boundary_sides = sum(score > 0.08 for score in boundary_scores)
    supported_boundary_sides = sum(score > 0.015 for score in boundary_scores)

    # A hand can hide much of one paper side, but the remaining sides should
    # still show a repeated transition from paper on the inside to background
    # on the outside
    if strong_boundary_sides < 2:
        return None

    if supported_boundary_sides < 3:
        return None

    if sorted_boundary_scores[2] < -0.04:
        return None

    boundary_contrast = float(np.mean(sorted_boundary_scores[:3]))

    texture = _visible_corner_interior_texture(
        gradient_magnitude=texture_gradient_magnitude,
        edge_map=texture_edges,
        quad_mask=quad_mask
    )

    # Carpet and fabric can produce many strong corners and even A4 like
    # geometry, but their interiors remain much more textured than paper
    if texture["edge_density"] > 0.055:
        return None

    if texture["flat_interior_ratio"] < 0.50:
        return None

    rect = cv2.minAreaRect(corners.astype("float32"))
    angle = rect[2]

    area_reward = 4.0 * min(geometry["area_ratio"] / 0.35, 1.0)

    score = (
        geometry["score"] +
        (7.0 * paper_fill) +
        (24.0 * boundary_contrast) +
        (2.0 * corner_support) +
        side_support +
        area_reward +
        (4.0 * texture["flat_interior_ratio"])
    )

    return {
        "score": float(score),
        "corners": corners,
        "ratio": float(geometry["ratio"]),
        "angle": float(angle),
        "area": float(geometry["area"]),
        "area_ratio": float(geometry["area_ratio"]),
        "paper_fill": float(paper_fill),
        "corner_support": float(corner_support),
        "side_support": float(side_support),
        "boundary_contrast": float(boundary_contrast),
        "supported_boundary_sides": int(supported_boundary_sides),
        "flat_interior_ratio": float(texture["flat_interior_ratio"]),
        "edge_density": float(texture["edge_density"]),
    }


def _create_continuous_paper_likelihood(image):
    """
    NEW: Build a continuous 0..1 paper likelihood image

    This is used for comparing a thin strip just inside a proposed paper edge
    with a thin strip just outside it. A real paper boundary should generally
    have a higher score on the inside.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype("float32")
    value = hsv[:, :, 2].astype("float32")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = cv2.GaussianBlur(
        lab[:, :, 0],
        (5, 5),
        0
    ).astype("float32")

    lightness_50, lightness_90 = np.percentile(lightness, [50, 90])
    value_50, value_90 = np.percentile(value, [50, 90])

    lightness_score = np.clip(
        (lightness - (lightness_50 - 10.0)) /
        max(20.0, (lightness_90 - lightness_50) + 10.0),
        0.0,
        1.0
    )

    value_score = np.clip(
        (value - (value_50 - 10.0)) /
        max(20.0, (value_90 - value_50) + 10.0),
        0.0,
        1.0
    )

    neutral_score = np.clip(
        (150.0 - saturation) / 120.0,
        0.0,
        1.0
    )

    brightness_score = np.maximum(lightness_score, value_score)

    return brightness_score * (0.55 + (0.45 * neutral_score))


def _paper_boundary_contrast_scores(paper_likelihood, corners):
    """
    NEW: Compare paper likelihood just inside and outside every proposed side.

    order_corners() returns tl, tr, br, bl. In image coordinates that order is
    clockwise, so (-dy, dx) points into the quadrilateral.
    """

    h, w = paper_likelihood.shape[:2]
    min_dim = min(h, w)
    offset = max(3, int(min_dim * 0.012))

    side_scores = []

    for i in range(4):
        p1 = corners[i].astype("float32")
        p2 = corners[(i + 1) % 4].astype("float32")

        direction = p2 - p1
        length = float(np.linalg.norm(direction))

        if length < 5.0:
            side_scores.append(-1.0)
            continue

        inward_normal = np.array(
            [-direction[1], direction[0]],
            dtype="float32"
        ) / length

        sample_count = max(20, int(length / 8.0))
        positions = np.linspace(
            0.08,
            0.92,
            sample_count,
            dtype="float32"
        )

        side_points = p1[None, :] + positions[:, None] * direction[None, :]

        inside_samples = []
        outside_samples = []

        for multiplier in (0.5, 1.0, 1.5):
            inside_points = side_points + inward_normal * (offset * multiplier)
            outside_points = side_points - inward_normal * (offset * multiplier)

            inside_x = np.clip(
                np.round(inside_points[:, 0]).astype("int32"),
                0,
                w - 1
            )
            inside_y = np.clip(
                np.round(inside_points[:, 1]).astype("int32"),
                0,
                h - 1
            )

            outside_x = np.clip(
                np.round(outside_points[:, 0]).astype("int32"),
                0,
                w - 1
            )
            outside_y = np.clip(
                np.round(outside_points[:, 1]).astype("int32"),
                0,
                h - 1
            )

            inside_samples.append(paper_likelihood[inside_y, inside_x])
            outside_samples.append(paper_likelihood[outside_y, outside_x])

        inside_values = np.mean(np.stack(inside_samples), axis=0)
        outside_values = np.mean(np.stack(outside_samples), axis=0)

        difference = inside_values - outside_values

        # Median prevents one reflection or one occluded segment from dominating
        # The positive only mean rewards repeated useful support along the side
        score = (
            (0.60 * float(np.median(difference))) +
            (0.40 * float(np.mean(np.clip(difference, 0.0, 1.0))))
        )

        side_scores.append(score)

    return side_scores


def _visible_corner_interior_texture(gradient_magnitude, edge_map, quad_mask):
    """
    NEW: Measure whether the proposed interior is mostly smooth like paper

    The Sobel gradient map and Canny edge map are precomputed once by the
    fallback detector and reused for every candidate
    """

    inside = quad_mask > 0
    gradient_values = gradient_magnitude[inside]

    if gradient_values.size == 0:
        return {
            "edge_density": 1.0,
            "flat_interior_ratio": 0.0,
        }

    edge_density = float(np.mean(edge_map[inside] > 0))

    # Gradient below 20 is a reasonably flat pixel at this working resolution
    flat_interior_ratio = float(np.mean(gradient_values < 20.0))

    return {
        "edge_density": edge_density,
        "flat_interior_ratio": flat_interior_ratio,
    }

def _corner_angle_degrees(point_a, point_b, point_c):
    """
    NEW: Return angle ABC in degrees
    """

    vector_1 = point_a.astype("float32") - point_b.astype("float32")
    vector_2 = point_c.astype("float32") - point_b.astype("float32")

    length_1 = float(np.linalg.norm(vector_1))
    length_2 = float(np.linalg.norm(vector_2))

    if length_1 == 0.0 or length_2 == 0.0:
        return 0.0

    cosine = float(
        np.clip(
            np.dot(vector_1, vector_2) / (length_1 * length_2),
            -1.0,
            1.0
        )
    )

    return float(np.degrees(np.arccos(cosine)))


def _detect_a4_by_paper_mask_fallback(image):
    """
    Fallback detector for images where a hand is placed on the A4 paper

    The normal detector looks for one clean rectangle contour
    That can fail when the hand breaks the paper surface and creates shadows

    This fallback instead tries to detect the bright paper region first,
    then estimates the A4 rectangle from the visible paper area

    Important limitation:
      The four A4 corners should still be visible for reliable warping
    """

    # image.shape is (height, width, channels) for a color image
    # We only need height and width here
    h, w = image.shape[:2]

    # Total image area in pixels
    # Used to compare candidate size relative to the whole photo
    image_area = h * w

    # Build two masks:
    # raw_mask keeps stricter white paper evidence
    # cleaned_mask reconnects paper regions broken by hand/shadow gaps
    raw_mask = _create_adaptive_paper_mask(image)
    cleaned_mask = _clean_paper_mask(raw_mask, image.shape[:2])

    # Find external contours from the estimated paper mask
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours by area from largest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Track the best fallback candidate found so far
    best_score = 0
    best_result = None

    # Only check the largest few regions.
    # Smaller regions are usually highlights, fingers, shadows, or table texture
    for contour in contours[:8]:
        visible_area = cv2.contourArea(contour)

        # With a hand on paper, visible paper may be smaller than empty A4,
        # but it should still be a meaningful part of the image
        if visible_area < image_area * 0.025:
            continue

        # Reject regions that are almost the whole image
        # This prevents the fallback from selecting a bright wall/table/background
        if visible_area > image_area * 0.90:
            continue

        # Create a convex hull around the visible paper region
        # This helps infer the full paper boundary when the hand interrupts the interior
        hull = cv2.convexHull(contour)

        # Create multiple corner candidates from the same visible paper region
        # approxPolyDP is preferred when it finds four real corners
        # minAreaRect is kept only as a backup candidate
        quad_candidates = _quad_candidates_from_hull(hull)

        for corners, source in quad_candidates:
            scored = _score_paper_quad(
                image_shape=image.shape[:2],
                raw_mask=raw_mask,
                cleaned_mask=cleaned_mask,
                corners=corners,
                source=source
            )

            if scored is None:
                continue

            if scored["score"] > best_score:
                best_score = scored["score"]
                best_result = scored

    if best_result:
        return {
            "a4_found": True,
            "approx_ratio": float(best_result["ratio"]),
            "confidence_score": float(best_result["score"]),
            "angle": float(best_result["angle"]),
            "corners_px": best_result["corners"].tolist(),
            "area": float(best_result["area"]),
            "image_area": float(image_area),
            "area_ratio": float(best_result["area_ratio"]),
            "visible_paper_ratio": float(best_result["paper_fill"]),
            "corner_support": float(best_result["corner_support"]),
            "side_support": float(best_result["side_support"]),
            "detection_method": f"paper_mask_fallback_{best_result['source']}",
        }

    return {"a4_found": False}


def _create_adaptive_paper_mask(image):
    """
    Create a whitepaper likelihood mask using image related brightness and saturation

    The goal is not simply "brightness > fixed number"
    Instead, the paper is treated as one of the brightest low saturation regions
    in the current image
    """

    # Convert to HSV because white paper usually has low saturation and high value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Convert to LAB because the L channel is a useful lightness estimate
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    # Blur lightness slightly so tiny texture/noise does not dominate the threshold
    lightness_blurred = cv2.GaussianBlur(lightness, (5, 5), 0)

    # Adaptive brightness thresholds
    # The paper should usually be in the brighter part of the image,
    # but shadowed paper should not be lost completely
    lightness_70 = np.percentile(lightness_blurred, 70)
    lightness_82 = np.percentile(lightness_blurred, 82)
    value_70 = np.percentile(value, 70)

    # Adaptive saturation threshold
    # White/grey paper is low saturation, whereas skin and backgrounds are morelikely to be higher in saturation
    saturation_55 = np.percentile(saturation, 55)
    saturation_limit = int(np.clip(saturation_55 + 25, 55, 125))

    # Keep reasonably bright, lowsaturation pixels
    # The second condition is stricter in brightness but more tolerant in saturation,
    # which helps preserve paper under warm lighting or mild shadows
    paper_like = (
        ((lightness_blurred >= max(105, lightness_70 - 5)) & (saturation <= saturation_limit)) |
        ((lightness_blurred >= max(125, lightness_82 - 8)) & (saturation <= 145)) |
        ((value >= max(115, value_70 - 5)) & (saturation <= saturation_limit))
    )

    return np.where(paper_like, 255, 0).astype("uint8")


def _clean_paper_mask(mask, image_shape):
    """
    Clean and reconnect the adaptive paper mask

    The hand can break the visible paper into several regions
    Morphological closing reconnects nearby paper areas without relying on a full clean contour
    """

    h, w = image_shape
    min_dim = min(h, w)

    # Remove small isolated white noise
    small_kernel_size = max(5, int(min_dim * 0.006))
    if small_kernel_size % 2 == 0:
        small_kernel_size += 1

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (small_kernel_size, small_kernel_size)
    )

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # Close medium gaps caused by shadows, finger edges, and uneven lighting
    medium_kernel_size = max(13, int(min_dim * 0.018))
    if medium_kernel_size % 2 == 0:
        medium_kernel_size += 1

    medium_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (medium_kernel_size, medium_kernel_size)
    )

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, medium_kernel, iterations=2)

    # Close larger handcreated interruptions, but keep it conservative
    # Too large a kernel can grow the paper into the background
    large_kernel_size = max(21, int(min_dim * 0.035))
    if large_kernel_size % 2 == 0:
        large_kernel_size += 1

    large_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (large_kernel_size, large_kernel_size)
    )

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, large_kernel, iterations=1)

    return cleaned


def _corners_from_contour_or_rect(contour, rect, image_area):
    """
    Prefer real contour corners when available
    Fall back to minAreaRect when the contour does not simplify cleanly to four corners
    """

    if contour is not None:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)

        if perimeter > 0:
            for epsilon_factor in (0.012, 0.018, 0.025, 0.035, 0.050):
                approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    ordered = order_corners(approx)

                    # Reject triangle like results caused by duplicated or
                    # collapsed corners before returning contour (shape) based corners
                    # This keeps tilted plain A4 images from producing a fake triangle
                    unique_corners = np.unique(ordered.astype("int32"), axis=0)

                    if len(unique_corners) == 4 and _quad_area(ordered) > image_area * 0.01:
                        return ordered

    # Convert the rotated rectangle into four corner points
    # boxPoints returns float coordinates
    box = cv2.boxPoints(rect)

    # Convert corner coordinates into integer pixel coordinates
    box = np.intp(box)  # aiemmin np.int0(box) = vanhentunut

    # Put the corners into the stable order needed for perspective correction
    return order_corners(box)


def _quad_candidates_from_hull(hull):
    """Build possible quadrilateral candidates from a papermask hull"""

    candidates = []
    perimeter = cv2.arcLength(hull, True)

    if perimeter > 0:
        # Try real polygon simplification first
        # This is usually more accurate than minAreaRect when the visible corners are clear
        for epsilon_factor in (0.010, 0.015, 0.020, 0.030, 0.045, 0.060):
            approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append((order_corners(approx), "approx"))
                break

    # Use extreme points as another lightweight estimate
    # This is useful when approxPolyDP gives more than four points because of minor mask abnormalities
    pts = hull.reshape(-1, 2)
    if len(pts) >= 4:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)

        extreme = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype="float32")

        if len(np.unique(extreme.astype("int32"), axis=0)) == 4:
            candidates.append((order_corners(extreme), "extreme"))

    # Keep minAreaRect as the final backup
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    candidates.append((order_corners(np.intp(box)), "rect"))

    return candidates


def _score_paper_quad(image_shape, raw_mask, cleaned_mask, corners, source):
    """
    Validate and score a fallback paper candidate.

    A candidate is accepted only when there is enough paper color support
    near the proposed A4 area, sides, and visible corners
    """

    h, w = image_shape
    image_area = h * w

    if not _is_valid_quad(corners, image_shape):
        return None

    area = _quad_area(corners)
    area_ratio = area / image_area

    # Reject tiny or almost whole image coering areas
    if area_ratio < 0.045 or area_ratio > 0.88:
        return None

    ratio = _quad_side_ratio(corners)
    ratio_error = abs(ratio - A4_RATIO)

    # Fallback allows more perspective/mask distortion than the clean contour detector,
    # but still rejects shapes that are too far from A4
    if ratio_error > 0.42:
        return None

    quad_mask = np.zeros((h, w), dtype="uint8")
    cv2.fillConvexPoly(quad_mask, corners.astype("int32"), 255)

    quad_pixels = cv2.countNonZero(quad_mask)
    if quad_pixels == 0:
        return None

    paper_inside = cv2.countNonZero(cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=quad_mask))
    paper_fill = paper_inside / quad_pixels

    # With a hand on the sheet, the paper fill will not be close to 1.0
    # But if it is too low, the candidate is probably not the A4 paper
    if paper_fill < 0.30:
        return None

    corner_support, min_corner_support = _corner_support_score(raw_mask, corners)

    # Because the working assumption is that all four A4 corners are visible,
    # every corner should have at least some whitepaper evidence nearby
    if min_corner_support < 0.025 or corner_support < 0.075:
        return None

    side_support, min_side_support = _side_support_score(raw_mask, corners)

    # At least some parts of the paper sides should be supported by the paper color mask
    if min_side_support < 0.025 or side_support < 0.070:
        return None

    # Use minAreaRect only when it is clearly supported
    # This prevents the fallback from accepting oversized rectangles too easily
    if source == "rect" and (corner_support < 0.095 or paper_fill < 0.36):
        return None

    rect = cv2.minAreaRect(corners.astype("float32"))
    angle = rect[2]

    # Score the candidate
    # The score is intentionally not calibrated probability
    # It is only used to compare candidates within the same image
    support_score = (0.50 * paper_fill) + (0.30 * corner_support) + (0.20 * side_support)
    source_bonus = {
        "approx": 1.12,
        "extreme": 1.03,
        "rect": 0.92,
    }.get(source, 1.0)

    score = area * support_score * source_bonus / (ratio_error + 0.08)

    return {
        "score": score,
        "corners": corners,
        "ratio": ratio,
        "angle": angle,
        "area": area,
        "area_ratio": area_ratio,
        "paper_fill": paper_fill,
        "corner_support": corner_support,
        "side_support": side_support,
        "source": source,
    }


def _corner_support_score(mask, corners):
    """Measure how much whitepaper evidence exists near each proposed corner"""

    h, w = mask.shape[:2]
    radius = max(8, int(min(h, w) * 0.018))

    scores = []

    for point in corners:
        x, y = int(round(point[0])), int(round(point[1]))

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius + 1)
        y2 = min(h, y + radius + 1)

        patch = mask[y1:y2, x1:x2]

        if patch.size == 0:
            scores.append(0.0)
            continue

        scores.append(cv2.countNonZero(patch) / patch.size)

    return float(np.mean(scores)), float(np.min(scores))


def _side_support_score(mask, corners):
    """Measure how much whitepaper evidence exists along the proposed A4 sides"""

    h, w = mask.shape[:2]
    thickness = max(6, int(min(h, w) * 0.010))

    side_scores = []

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]

        line_mask = np.zeros((h, w), dtype="uint8")
        cv2.line(
            line_mask,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            255,
            thickness
        )

        line_pixels = cv2.countNonZero(line_mask)
        if line_pixels == 0:
            side_scores.append(0.0)
            continue

        supported_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=line_mask))
        side_scores.append(supported_pixels / line_pixels)

    return float(np.mean(side_scores)), float(np.min(side_scores))


def _is_valid_quad(corners, image_shape):
    """Validate that the corners form usable convex quadrilateral inside the image"""

    h, w = image_shape
    corners = corners.astype("float32")

    if corners.shape != (4, 2):
        return False

    if not np.all(np.isfinite(corners)):
        return False

    # Allow a very small margin around the image boundary,
    # because detected paper corners can be exactly at the edge of the photo
    margin = 3
    if np.any(corners[:, 0] < -margin) or np.any(corners[:, 0] > w + margin):
        return False

    if np.any(corners[:, 1] < -margin) or np.any(corners[:, 1] > h + margin):
        return False

    contour = corners.reshape(4, 1, 2)
    if not cv2.isContourConvex(contour):
        return False

    if _quad_area(corners) <= 0:
        return False

    return True


def _quad_area(corners):
    """Calculate quadrilateral area in pixels"""

    return abs(cv2.contourArea(corners.astype("float32").reshape(4, 1, 2)))


def _quad_side_ratio(corners):
    """
    Calculate A4 like side ratio from ordered corners
    Works for both portrait and landscape orientation
    """

    tl, tr, br, bl = corners.astype("float32")

    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)

    avg_width = (top + bottom) / 2.0
    avg_height = (left + right) / 2.0

    shorter = min(avg_width, avg_height)
    longer = max(avg_width, avg_height)

    if shorter == 0:
        return 0.0

    return longer / shorter