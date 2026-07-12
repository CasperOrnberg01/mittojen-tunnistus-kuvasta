# app/services/quality_check.py
# Image quality assessment logic.
# This file checks whether an uploaded image is likely good enough for A4 detection.

import cv2
import numpy as np


# Minimum sharpness score accepted by the blur check
# The score comes from Laplacian variance: 
#   - high value = sharper image
#   - low value = blurrier image
# 80.0 is a practical starting threshold for phone photos
BLUR_THRESHOLD       = 80.0   # Laplacian variance - below this = blurry

# Minimum average grayscale brightness.
# Grayscale brightness uses a range from 0 to 255:
# 0 = black
# 255 = white
BRIGHTNESS_MIN       = 50     # 50 or under = possibly too dark for reliable detection

# Maximum average grayscale brightness
# Overexposed images can lose paper edges because white areas blend together
BRIGHTNESS_MAX       = 220    # 220 or more = perhaps too bright

# Maximum allowed estimated camera tilt in degrees
# 30 degrees is permissive enough for normal handheld photos,
# but still warns when the camera angle may distort the paper too much.
TILT_MAX_DEG         = 30.0   # Maximum camera angle in respect to horizontal


def check_blur(gray):
    """
    Estimate image blur using Laplacian variance.

    The Laplacian operator reacts strongly to edges and fine details
    Sharp images have clear edges so= Laplacian variance is higher
    Blurry images have weak edges so = Laplacian variance is lower
    """

    # Apply the Laplacian operator to the grayscale image
    # cv2.CV_64F allows negative and high precision values during calculation
    # .var() calculates variance: stronger edge changes produce a higher value
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Return both the numeric score and a human readable message
    return {
        "blur_score": round(float(lap_var), 2),
        "blur_ok":    lap_var >= BLUR_THRESHOLD,
        "blur_msg":   "ok" if lap_var >= BLUR_THRESHOLD else f"Image may be blurry (score {lap_var:.0f} < {BLUR_THRESHOLD})"
    }


def check_brightness(gray):
    """
    Check average image brightness.

    Very dark or very bright images can make edge detection unreliable.
    A4 paper detection depends heavily on finding the paper border.
    """

    # Calculate the average grayscale value of the whole image
    # Result is between 0-255
    mean_val = float(np.mean(gray))

    # Brightness is acceptable when it is inside the allowed range
    ok = BRIGHTNESS_MIN <= mean_val <= BRIGHTNESS_MAX

    # Create a clear warning message if the image is too dark or too bright
    if mean_val < BRIGHTNESS_MIN:
        msg = f"Image too dark (brightness {mean_val:.0f} < {BRIGHTNESS_MIN})"
    elif mean_val > BRIGHTNESS_MAX:
        msg = f"Image too bright / overexposed (brightness {mean_val:.0f} > {BRIGHTNESS_MAX})"
    else:
        msg = "ok"

    # Return brightness details for API responses and debug overlays
    return {
        "brightness":    round(mean_val, 1),
        "brightness_ok": ok,
        "brightness_msg": msg
    }



def check_tilt(image):
    """
    Estimate camera/image tilt using detected straight lines.

    This is only an approximation.
    It uses Hough line detection to find strong lines in the image,
    then estimates how far those lines are from horizontal or vertical.
    """

    # Convert the image to grayscale because Canny edge detection uses one channel
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges before looking for lines
    # 50 and 150 are the lower and upper Canny thresholds
    edges   = cv2.Canny(gray, 50, 150)

    # Detect line segments using the probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,

        # Distance resolution in pixels.
        # 1 means the algorithm checks line distances at 1-pixel precision
        1,

        # Angle resolution in radians
        # np.pi / 180 means 1-degree precision
        np.pi / 180,

        # Minimum votes needed to accept a line
        # 80 filters out many weak/random texture lines
        threshold=80,

        # Minimum line length in pixels
        # 60 ignores short noise lines but keeps meaningful paper/background edges
        minLineLength=60,

        # Maximum gap in pixels allowed between line segments that should be connected
        # 10 helps bridge small breaks in edges
        maxLineGap=10
    )


    # If no lines are found, do not fail the image
    # No lines may simply mean the image has weak edges, and the blur/brightness checks
    if lines is None:
        return {"tilt_deg": 0.0, "tilt_ok": True, "tilt_msg": "No lines found (assuming ok)"}

    # Store all detected line angles here
    angles = []

    # Loop through every detected line segment
    for line in lines:
        # HoughLinesP returns each line as [[x1, y1, x2, y2]]
        pt = line[0]

        # If point is a single number, skip it
        if isinstance(pt, (np.int32, np.int64, int, float)):
            return {"tilt_deg": 0.0, "tilt_ok": True}

        # If point has less than 4 coordinates, skip it
        if len(pt) < 4:
            return {"tilt_deg": 0.0, "tilt_ok": True}

        x1, y1, x2, y2 = pt


        # Calculate the line angle in degrees
        # arctan2 gives the angle of the line direction
        # abs() makes negative angles positive
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


         # Normalize angles into the 0-90 degree range
        # Example: 120 degrees becomes 60 degrees because it is the same line direction
        if angle > 90:
            angle = 180 - angle

        # Save the normalized line angle
        angles.append(angle)

    
    # Use the median angle instead of the average
    # Median is more robust when a few wrong/noisy lines are detected
    tilt = float(np.median(angles))


    # Convert the line angle into "tilt away from horizontal or vertical"
    # A paper edge can be horizontal (0 degrees) or vertical (90 degrees)
    # We use the closest of those two directions as the reference
    if tilt < 45:
        tilt_from_horizontal = abs(tilt - 0)
    else:
        tilt_from_horizontal = abs(90 - tilt)

    # Check whether the estimated tilt is within the allowed limit
    ok = tilt_from_horizontal <= TILT_MAX_DEG

    # Build a readable result message
    msg = "ok" if ok else f"Camera angle too steep ({tilt_from_horizontal:.1f} deg > {TILT_MAX_DEG})"

    # Return the tilt estimate and status
    return {
        "tilt_deg": round(tilt_from_horizontal, 1),
        "tilt_ok": ok,
        "tilt_msg": msg
    }


def assess_image_quality(image):
    """
    Run all image quality checks and return a combined summary.

    The function does not directly reject images
    It returns warnings so the API/debug overlay can show quality problems while still
    allowing A4 detection to run.
    """

    # Convert once to grayscale for blur and brightness checks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run each individual quality check
    blur_result       = check_blur(gray)
    brightness_result = check_brightness(gray)
    tilt_result       = check_tilt(image)

    # Collect warning messages from failed checks
    warnings = []

    # Add blur warning if needed
    if not blur_result["blur_ok"]:
        warnings.append(blur_result["blur_msg"])

    # Add brightness warning if needed
    if not brightness_result["brightness_ok"]:
        warnings.append(brightness_result["brightness_msg"])

    # Add tilt warning if needed
    if not tilt_result["tilt_ok"]:
        warnings.append(tilt_result["tilt_msg"])


    # Accept the image if it has at most one quality warning
    # This is intentionally not too strict becauose rejecting too many usable images = weak quality can still be shown as a warnin in the debug overlay
    acceptable = len(warnings) <= 1
    
    # Merge all results into one dictionary
    # **dict expands key/value pairs into the returned dictionary
    return {
        "acceptable":     acceptable,
        "warnings":       warnings,
        **blur_result,
        **brightness_result,
        **tilt_result,
    }