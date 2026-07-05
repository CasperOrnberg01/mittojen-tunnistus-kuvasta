# app/routes/upload_warp_debug.py
# Warp debug endpoint that detects A4 paper and returns straightened A4 image

import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException  # FastAPI tools for routing, file uploads, and proper API errors
from fastapi.responses import Response  # Response lets this route return JPEG bytes instead of JSON

from app.services.a4_detection import detect_a4  # detect_a4 finds the paper and returns its corner coordinates

# warp_a4 performs perspective correction
# get_scale_px_per_mm returns useful measurement scale meta data
from app.services.a4_warp import warp_a4, get_scale_px_per_mm, get_a4_orientation 

from app.services.debug_overlay import encode_image_jpeg # encode_image_jpeg converts the warped OpenCV image into browser readable JPEG bytes

# Create a router for this file
router = APIRouter()

# Create a POST endpoint at /upload/warp-debug
# responses documents the output as image/jpeg in Swagger
# response_class=Response prevents FastAPI from converting result to JSON
@router.post(
    "/upload/warp-debug",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def upload_warp_debug(file: UploadFile = File(...)):
    """
    Upload image -> detect A4 -> return warped A4 image as JPEG.

    Use this in Swagger:
      POST /upload/warp-debug
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

    # Empty files cannot be decoded or warped
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Convert bytes into a NumPy byte array
    np_array = np.frombuffer(contents, np.uint8)

    # Decode the byte array into an OpenCV BGR image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    # OpenCV returns None if decoding fails
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")


    # First detect A4 paper in the original image
    # Warping only makes sense if we have reliable corner coordinates
    a4_result = detect_a4(image)
    
    # Stop if A4 was not found or corners are missing
    # 422 is used because image was valid, but it does not contain usable A4 data
    if not a4_result.get("a4_found") or "corners_px" not in a4_result:
        raise HTTPException(
            status_code=422,
            detail="A4 paper not found, cannot warp image"
        )
    

    # Detect whether the found A4 is portrait or landscape
    # This keeps response headers consistent with the warped output
    orientation = get_a4_orientation(a4_result["corners_px"])

    # Perspective correct the detected A4 into a fixed size rectangle
    # The matrix is returned for future use, even though this endpoint only displays image
    try:
        warped, matrix = warp_a4(image, a4_result["corners_px"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Perspective correction failed: {str(e)}"
        )
    

    # Convert the warped image into JPEG bytes for Swagger/browser display
    jpeg_bytes = encode_image_jpeg(warped)
 
    # Calculate measurement scale metadata for response headers
    # This confirms how many pixels represent one milimeter after warping
    scale = get_scale_px_per_mm(orientation)


    # Return the warped A4 image directly
    # Headers expose useful metadata to clients without changing the image response body
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "inline; filename=warped_a4.jpg",
            "X-A4-Found": "True",
            "X-A4-Orientation": scale["orientation"],
            "X-Warped-Width-PX": str(scale["warped_width_px"]),
            "X-Warped-Height-PX": str(scale["warped_height_px"]),
            "X-PX-Per-MM-AVG": str(scale["px_per_mm_avg"]),
        }
    )