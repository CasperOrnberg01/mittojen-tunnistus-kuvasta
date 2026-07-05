# app/services/a4_detection.py
# A4 paper sheet detection logic
# this file tries to find an A4 sized rectangle from an uploaded image


import cv2
import numpy as np


# A4 aspect ratio reference 297mm / 210mm = 1.414...
# A4 measures: 210x297mm
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

    # CHANGED: The old sum/difference method could select the same corner twice
    # when the A4 sheet was strongly rotated or diamond-shaped in the image.
    # That caused triangle-looking debug outlines even when the paper was detected.
    #
    # The safer approach is:
    # 1. Find the center of the four points.
    # 2. Sort all points around that center by angle.
    # 3. Rotate the ordered list so the first point is the image-space top-left corner.
    #
    # This preserves the correct polygon order without duplicating corners.
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # CHANGED: Start from the most top-left-like point.
    # Using x + y keeps the output compatible with the rest of the warp code,
    # but it is now only used to choose the start point, not to identify every corner.
    start_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start_index, axis=0)

    # CHANGED: Make sure the points are in [tl, tr, br, bl] order, not reversed.
    # In image coordinates, y grows downward. For the expected order, the signed
    # polygon area should be positive. If it is negative, reverse the direction
    # while keeping the same first corner.
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
    # clipLimit=2.0 limits how aggressively contrast is boosted
    # A value around 2.0 is a common safe starting point: enough to reveal paper edges,
    # but not so strong that it creates too much noise
    # tileGridSize=(8, 8) splits the image into small regions for local contrast
    # 8x8 is commonly used because it balances local detail and smoothness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur image for noise reduction
    # GaussianBlur smooth image and removes small noise
    # kernel size set to (5,5)   must be odd numbers for the best results
    # standard deviation 0 = Lets OpenCV automatically calculate the best standard deviation based on the kernel size
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges with the Canny algorithm.
    # 50 is the lower threshold and 150 is the upper threshold.
    # Edges stronger than 150 are definitely kept
    # Edges between 50 and 150 are kept if connected to stronger edges
    edges = cv2.Canny(blurred, 50, 150)

    # Create a rectangular morphology kernel
    # (5, 5) means small gaps up to a few pixels can be closed.
    # This helps turn broken paper-edge segments into more complete contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Close small gaps in the detected edges.
    # MORPH_CLOSE performs dilation followed by erosion.
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

    # Track the best candidate found so far
    # A score of 0 means no candidate has been accepted yet
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
        # 0.01 = 1% of the whole image
        # Anything smaller is unlikely to be the full A4 sheet and is often texture or some kind of noise
        if area < image_area * 0.01:
            continue

        # Calculate how much of the whole image this contour covers.
        # Example: 0.50 means the contour covers about 50% of the image.
        area_ratio = area / image_area

        # Reject contours that are almost the whole image
        # 0.85 means 85% of the image
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

        # Create the convex hull around the contour
        # The hull is the smallest convex shape that contains the contour
        hull = cv2.convexHull(contour)

        # Calculate hull area in pixels
        hull_area = cv2.contourArea(hull)

        # Avoid division by zero
        if hull_area == 0:
            continue

        # Solidity measures how much the contour fills its convex hull
        # A clean paper rectangle should be solid and close to 1.0
        solidity = area / hull_area

        # Reject contours that are too irregular or broken
        # 0.92 means the contour must fill at least 92% of its convex hull
        # This keeps the detector focused on rectangle-like paper shapes
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

        # Reject candidates that are too edge-heavy inside
        # 0.85 is very permissive, but helps remove noisy regions that are mostly texture
        if edge_density > 0.85:
            continue

        # Score the candidate
        # Larger area is better
        # Higher solidity is better
        # Smaller aspect-ratio error is better
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
        # Try to estimate real contour corners first.
        # If that is not reliable, fall back to minAreaRect corners.
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

    # If the normal contour detector fails, try a fallback designed for A4 images with a hand on top
    # This does not replace the main detector
    # It only runs when the clean contourbased method cannot find a reliable A4 candidate
    fallback_result = _detect_a4_by_paper_mask_fallback(image)

    if fallback_result.get("a4_found", False):
        return fallback_result

    # No candidate passed all filters
    return {
        "a4_found": False,
        "needs_manual_corners": True,
        "reason": "No reliable A4 candidate found"
    }



def _detect_a4_by_paper_mask_fallback(image):
    """
    Fallback detector for images where a hand is placed on the A4 paper.

    The normal detector looks for one clean rectangle contour.
    That can fail when the hand breaks the paper surface and creates shadows.

    This fallback instead tries to detect the bright paper region first,
    then estimates the A4 rectangle from the visible paper area.

    Important limitation:
      The four A4 corners should still be visible for reliable warping.
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
    Create a white-paper likelihood mask using image-relative brightness and saturation.

    The goal is not simply "brightness > fixed number".
    Instead, the paper is treated as one of the brightest low-saturation regions
    in the current image.
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

    # Adaptive brightness thresholds.
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
    Clean and reconnect the adaptive paper mask.

    The hand can break the visible paper into several regions.
    Morphological closing reconnects nearby paper areas without relying on a full clean contour.
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

                    # CHANGED: Reject triangle-like results caused by duplicated or
                    # collapsed corners before returning contour-based corners.
                    # This keeps tilted plain A4 images from producing a fake triangle.
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
    """
    Build possible quadrilateral candidates from a paper-mask hull.
    """

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

    A candidate is accepted only when there is enough paper-color support
    near the proposed A4 area, sides, and visible corners.
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
    # every corner should have at least some white-paper evidence nearby
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
    """
    Measure how much whitepaper evidence exists near each proposed corner
    """

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
    """
    Measure how much whitepaper evidence exists along the proposed A4 sides
    """

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
    """
    Validate that the corners form a usable convex quadrilateral inside the image.
    """

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
    """
    Calculate quadrilateral area in pixels.
    """

    return abs(cv2.contourArea(corners.astype("float32").reshape(4, 1, 2)))



def _quad_side_ratio(corners):
    """
    Calculate A4-like side ratio from ordered corners.
    Works for both portrait and landscape orientation.
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
