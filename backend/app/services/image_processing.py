# app/services/image_processing.py
# Process the picture with OpenCV
# image processing "logic" here
# Notes: Removed duplicate helper code and now imports and reuses debug_overlay.py and a4_warp.py helper modules

import cv2
import numpy as np

import base64 # base64 converts debug image bytes into a text string that can be returned in JSON
from app.services.a4_detection import detect_a4 # detect_a4 finds A4 paper and returns detection data such as corners and confidence

# Reuse the shared debug overlay helper instead of keeping another local overlay function here
# This keeps all visual A4 overlay drawing in one place: debug_overlay.py
from app.services.debug_overlay import draw_a4_overlay, encode_image_jpeg

# Reuse the shared A4 warp module instead of duplicating A4 constants and warp code here
# This keeps perspective correction and scale values in one place: a4_warp.py
from app.services.a4_warp import (warp_a4, get_scale_px_per_mm, get_a4_orientation,)


def process_image(image_bytes):
    """
    Decode uploaded image bytes, detect A4, optionally warp it, and return JSON data
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

    # Create a visual debug image using the shared overlay helper
    # This avoids duplicating A4 drawing code inside this file
    debug_image = draw_a4_overlay(image, a4_result)
    

    # Encode the debug image as JPEG
    # This allows the debug image to be included in JSON as base64 text
    debug_jpeg_bytes = encode_image_jpeg(debug_image)

    # Convert JPEG bytes into a base64 string
    # Frontend code can display this as: data:image/jpeg;base64,<string>
    debug_base64 = base64.b64encode(debug_jpeg_bytes).decode("utf-8")
   
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
            # Detect whether the found A4 is portrait or landscape.
            # This keeps JSON metadata consistent with the warped output.
            orientation = get_a4_orientation(corners)

            # Warp the original image into a normalized A4 rectangle
            warped, M = warp_a4(image, corners)

            # Calculate the pixels per millimeter scale from shared warp module
            scale = get_scale_px_per_mm(orientation)

            # Build metadata about warped A4 image
            warp_info = {
                "orientation": scale["orientation"],
                "warped_size_px": {
                    "width": scale["warped_width_px"],
                    "height": scale["warped_height_px"]
                },
                "scale_px_per_mm": scale["px_per_mm_avg"],
                "scale_px_per_mm_x": scale["px_per_mm_x"],
                "scale_px_per_mm_y": scale["px_per_mm_y"],
                "a4_real_mm": {
                    "width": scale["a4_width_mm"],
                    "height": scale["a4_height_mm"]
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
