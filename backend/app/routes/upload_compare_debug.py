# app/routes/upload_compare_debug.py
# Compare debug endpoint
# This endpoint returns original A4 detection view and warped A4 view side by side

import cv2 # cv2 is OpenCV, used here for image decoding, resizing, text drawing, and image combining
import numpy as np # numpy is used to convert raw upload bytes into arrays that OpenCV can read and combine

from fastapi import APIRouter, UploadFile, File, HTTPException  # FastAPI route and upload tools
from fastapi.responses import Response  # Response lets us return raw JPEG bytes instead of JSON

from app.services.a4_detection import detect_a4  # detect_a4 finds the paper and returns its corner coordinates
from app.services.quality_check import assess_image_quality # assess_image_quality checks blur, brightness and estimated camera tilt angle

# draw_a4_overlay draws the original image debug view
# encode_image_jpeg converts the final combined image into browser readable JPEG bytes
from app.services.debug_overlay import draw_a4_overlay, encode_image_jpeg

# warp_a4 performs perspective correction
# get_scale_px_per_mm returns useful measurement scale metadata
from app.services.a4_warp import warp_a4, get_scale_px_per_mm, get_a4_orientation


# Create a router for this file
router = APIRouter()


# Resize image to a target height while keeping the original aspect ratio
# This is needed because the original photo and warped A4 image often have different sizes
# INTER_AREA is good when shrinking images, INTER_CUBIC is good when enlarging images
# Returns the resized image so it can be placed next to another image

def resize_to_height(image, target_height):
    original_height, original_width = image.shape[:2]
    scale = target_height / original_height
    target_width = int(original_width * scale)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


# Add a simple title bar above an image
# This makes the side byside debug output easier to read in swagger
#image itself is not overwritten because it is placed below the title bar

def add_title_bar(image, title):
    bar_height = 42
    title_bar = np.zeros((bar_height, image.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        title_bar,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    return np.vstack([title_bar, image])


# Create a placeholder image when A4 cannot be warped
# This lets the endpoint still return a useful visual debug image instead of only an error

def create_placeholder(width, height, lines):
    placeholder = np.full((height, width, 3), 35, dtype=np.uint8)

    y = 60
    for line in lines:
        cv2.putText(
            placeholder,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 36

    return placeholder


# Create a POST endpoint at /upload/compare-debug
# responses documents the output as image/jpeg in Swagger
# response_class=Response prevents FastAPI from converting the result to JSON
@router.post(
    "/upload/compare-debug",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def upload_compare_debug(file: UploadFile = File(...)):
    """
    Upload image -> detect A4 -> show original detection and warped A4 side by side.

    Use this in Swagger:
      POST /upload/compare-debug
    """

    # Supported image types
    # Notes: !!Keep this list consistent with the other upload endpoints!!
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}

    # Reject unsupported formats before decoding
    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # Read the uploaded image bytes
    contents = await file.read()

    # Empty files cannot be decoded or compared
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Convert bytes into a NumPy byte array
    np_array = np.frombuffer(contents, np.uint8)

    # Decode the byte array into an OpenCV BGR image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # OpenCV returns None if decoding fails
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")


    # Check image quality for the leftside overlay
    # This does not block A4 detection, it only adds useful debug information
    quality = assess_image_quality(image)

    # Detect A4 paper in the original image
    a4_result = detect_a4(image)

    # Create the left side image: original photo with detection overlay
    annotated = draw_a4_overlay(image, a4_result, quality_result=quality)


    # The warped A4 image will be placed on the right side
    # If A4 is not found, we show a placeholder instead
    a4_found = a4_result.get("a4_found", False) and "corners_px" in a4_result
    warp_ok = False

    # Default scale metadata is portrait.
    # This is used only for placeholders when A4 is not found or warping fails.
    scale = get_scale_px_per_mm("portrait")

    if a4_found:
        try:
            # Detect whether the found A4 is portrait or landscape
            # This keeps the debug metadata consistent with the warped output
            orientation = get_a4_orientation(a4_result["corners_px"])
            scale = get_scale_px_per_mm(orientation)

            # Create the rightside image: straightened A4 paper
            warped, matrix = warp_a4(image, a4_result["corners_px"])
            warp_ok = True
        except Exception as e:
            # If warping fails, keep the debug endpoint visual instead of crashing
            warped = create_placeholder(
                scale["warped_width_px"],
                scale["warped_height_px"],
                ["WARP FAILED", str(e)]
            )
    else:
        # Placeholder explains why the right side has no warped A4 result
        warped = create_placeholder(
            scale["warped_width_px"],
            scale["warped_height_px"],
            ["A4 NOT FOUND", "WARP UNAVAILABLE"]
        )


    # Resize the original debug image to match the warped A4 height
    # This creates a clean side-by-side comparison
    target_height = warped.shape[0]
    annotated_resized = resize_to_height(annotated, target_height)

    # Add titles above both sides so the output is easy to understand
    left_side = add_title_bar(annotated_resized, "ORIGINAL + DETECTION")
    right_side = add_title_bar(warped, "WARPED A4")

    # Add a small dark gap between the two images
    gap_width = 24
    gap = np.full((left_side.shape[0], gap_width, 3), 25, dtype=np.uint8)

    # Combine left image, gap, and right image horizontally
    combined = np.hstack([left_side, gap, right_side])

    # Convert the combined OpenCV image into JPEG bytes for swagger/browser display
    jpeg_bytes = encode_image_jpeg(combined)


    # Return the combined debug image directly
    # Headers expose useful metadata to clients without changing the image response body
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "inline; filename=compare_debug_a4.jpg",
            "X-A4-Found": str(a4_found),
            "X-Warp-OK": str(warp_ok),
            "X-A4-Orientation": scale["orientation"],
            "X-Warped-Width-PX": str(scale["warped_width_px"]),
            "X-Warped-Height-PX": str(scale["warped_height_px"]),
            "X-PX-Per-MM-AVG": str(scale["px_per_mm_avg"]),
        }
    )
