# app/services/a4_detection.py
# A4 paper sheet detection logic
# this file tries to find an A4 sized rectangle from an uploaded image


import cv2
import numpy as np


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


    # For image coordinates, x grows to the right and y grows downward
    # The top-left point usually has the smallest x + y value
    # The bottom-right point usually has the largest x + y value
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # np.diff calculates y - x for each point.
    # The top-right point usually has the smallest y - x value.
    # The bottom-left point usually has the largest y - x value.
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Return the ordered corners as float32 for later OpenCV operations
    return np.array([tl, tr, br, bl], dtype="float32")



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

    # A4 aspect ratio reference 297mm / 210mm = 1.414...
    # A4 measures: 210x297mm
    A4_RATIO = 297 / 210
    
    # Track the best candidate found so far
    # A score of 0 means no candidate has been accepted yet
    best_score = 0

    # Stores the metadata of the best A4 candidate
    best_candidate = None

    # Stores the rotated rectangle from cv2.minAreaRect for the best candidate
    # The rectangle is later converted into four corner points
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
            best_rect = rect  # note: storing rect for later use
            best_candidate = {
                "ratio": ratio,
                "angle": angle,
                "score": score,
                "solidity": solidity,
                "area": area,
                "image_area": image_area,
                "area_ratio": area_ratio,
            }
           

    # If a valid candidate was found, convert it into API-friendly output
    if best_candidate and best_rect is not None:
        # Convert the rotated rectangle into four corner points.
        # boxPoints returns float coordinates.
        box = cv2.boxPoints(best_rect)

        # Convert corner coordinates into integer pixel coordinates.
        box = np.intp(box)  # aiemmin np.int0(box) = vanhentunut

        # Put the corners into the stable order needed for perspective correction.
        ordered = order_corners(box)

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
        }

    # No candidate passed all filters
    return {"a4_found": False}
