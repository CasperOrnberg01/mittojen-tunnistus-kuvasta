# app/services/image_processing.py
# Process the picture with OpenCV
# image processing "logic" here

import cv2
import numpy as np

import base64 # base64 converts debug image bytes into a text string that can be returned in JSON.
from app.services.a4_detection import detect_a4 # detect_a4 finds A4 paper and returns detection data such as corners and confidence.


# Real A4 paper width in millimeters
A4_WIDTH_MM = 210.0

# Real A4 paper height in millimeters
A4_HEIGHT_MM = 297.0

# Output width for a normalized A4 image
# 794px is approximately 210mm at 96 DPI
A4_WIDTH_PX = 794    # 210mm * 3.78

# Output height for a normalized A4 image
# 1123px is approximately 297mm at 96 DPI
A4_HEIGHT_PX = 1123  # 297mm * 3.78



def draw_a4_overlay(image, corners):
    """
    Draw a simple A4 overlay for the JSON endpoint debug image

    Notes:
      There is a more complete overlay implementation in debug_overlay.py
      This helper is a simpler local version used by process_image()
    """

    # Copy the image so the original is not changed
    overlay = image.copy()

    # Convert corner coordinates into int32 pixel coordinates for OpenCV drawing
    pts = np.array(corners, dtype=np.int32)

    # Draw the A4 outline in green
    # True means the polygon is closed
    # (0, 255, 0) is green in OpenCV BGR color order
    # thickness=3 makes the box visible in the returned debug image
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)

    # Draw red points on each detected corner
    # Radius 10 makes the points visible
    # -1 fills the circles
    for (x, y) in pts:
        cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)

    # Return the image with the simple overlay
    return overlay



def warp_a4(image, corners):
    """
    Perspective-correct the detected A4 paper into a fixed rectangle

    corners must be ordered as:
    [top-left, top-right, bottom-right, bottom-left]

    Note:
      The newer separate file a4_warp.py contains the same idea in a cleaner module
      Long term, it would be better to use one shared warp implementation
    """

    # Destination coordinates for the final straight A4 image
    # -1 is used because pixel coordinates start at 0
    dst = np.array([
        [0,           0          ],
        [A4_WIDTH_PX - 1,  0          ],
        [A4_WIDTH_PX - 1,  A4_HEIGHT_PX - 1],
        [0,           A4_HEIGHT_PX - 1]
    ], dtype="float32")

    # Source coordinates are the detected A4 corners in the original image
    # float32 is required by OpenCV perspective transform functions
    src = np.array(corners, dtype="float32")

    # Calculate perspective transform matrix
    # This maps the original detected A4 into the fixed destination rectangle
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform
    # The output image size is width x height
    warped = cv2.warpPerspective(image, M, (A4_WIDTH_PX, A4_HEIGHT_PX))

    # Return both the warped image and the transform matrix
    return warped, M


def get_scale(warped_image):
    """
    Calculate the average pixels per millimeter scale for the warped A4 image
    """
    # Horizontal pixels per millimeter
    px_per_mm_x = A4_WIDTH_PX / A4_WIDTH_MM

    # Vertical pixels per millimeter
    px_per_mm_y = A4_HEIGHT_PX / A4_HEIGHT_MM

    # Average both directions for one simple scale value
    # For basic measurements this is convenient, but axis-specific scale may be better later
    px_per_mm = (px_per_mm_x + px_per_mm_y) / 2

    # Round to 4 decimals so the JSON output is readable but still precise enough
    return round(px_per_mm, 4)


def process_image(image_bytes):
    """
    Decode uploaded image bytes, detect A4, optionally warp it, and return JSON data.
    """

    # Convert raw bytes into a NumPy uint8 array
    # cv2.imdecode expects this format
    np_array = np.frombuffer(image_bytes, np.uint8)

    # Decode the byte array into an OpenCV BGR color image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Print the decoded image type to the server console for debugging
    print("DEBUG - image type:", type(image))

    # If decoding failed, OpenCV returns None
    if image is None:
        print("DEBUG - image is None")
        return {"error": "Invalid image or unsupported format"}

    # An image with size 0 is not usable
    if image.size == 0:
        return {"error": "Empty image"}

    # Read image height and width
    # shape is (height, width, channels)
    height, width = image.shape[:2]


    # Run A4 detection on the decoded image
    a4_result = detect_a4(image)

    # Create debug image copy
    # If A4 is found, the detected corners will be drawn on this copy
    debug_image = image.copy()

    # Draw the simple A4 overlay only when detection succeeded and corners exist
    if a4_result.get("a4_found") and "corners_px" in a4_result:
        debug_image = draw_a4_overlay(image, a4_result["corners_px"])
    

    # Encode the debug image as JPEG
    # This allows the debug image to be included in JSON as base64 text
    _, buffer = cv2.imencode(".jpg", debug_image)

    # Convert JPEG bytes into a base64 string
    # Frontend code can display this as: data:image/jpeg;base64,<string>
    debug_base64 = base64.b64encode(buffer).decode("utf-8")
   
    # Store perspective correction info here
    # It stays None when A4 is not found
    warp_info = None


    # Only warp when A4 detection succeeded and corners exist
    if a4_result.get("a4_found") and "corners_px" in a4_result:
        # Store corners in a local variable for readability
        corners = a4_result["corners_px"]

        # Try perspective correction
        # If warping fails, API should still return useful detection data
        try:
            # Warp the original image into a normalized A4 rectangle
            warped, M = warp_a4(image, corners)

            # Calculate the pixels-per-millimeter scale
            px_per_mm = get_scale(warped)

            # Build metadata about the warped A4 image
            warp_info = {
                "warped_size_px": {
                    "width": A4_WIDTH_PX,
                    "height": A4_HEIGHT_PX
                },
                "scale_px_per_mm": px_per_mm,
                "a4_real_mm": {
                    "width": A4_WIDTH_MM,
                    "height": A4_HEIGHT_MM
                }
            }


            # Mark that a warped version could be created
            # The actual warped image is not returned here, only metadata
            a4_result["warped_available"] = True

        except Exception as e:
            # Print the technical error to the server console
            print(f"DEBUG - warp failed: {e}")
            # Return the warp error as JSON metadata instead of crashing the request
            warp_info = {"error": f"Perspective correction failed: {str(e)}"}


    # Return the full JSON response
    # height and width describe the original uploaded image
    # a4_detection contains the detection result
    # perspective_correction contains warp metadata
    # debug_image_base64 contains a simple visual debug image
    return {
        "height": height,
        "width": width,
        "a4_detection": a4_result,
        "perspective_correction": warp_info,
        "debug_image_base64": debug_base64
    }
