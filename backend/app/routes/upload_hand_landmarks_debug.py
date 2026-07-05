# app/routes/upload_hand_landmarks_debug.py
# Debug endpoint for MediaPipe hand landmark detection
# Upload image -> detect A4 -> warp A4 -> run MediaPipe -> return landmarks debug image

import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from app.services.a4_detection import detect_a4
from app.services.a4_warp import warp_a4, get_a4_orientation, get_scale_px_per_mm
from app.services.debug_overlay import encode_image_jpeg
from app.services.hand_landmarks import (
    detect_hand_landmarks,
    draw_hand_landmarks_debug,
    estimate_hand_measurements,
)


router = APIRouter()


@router.post(
    "/upload/hand-landmarks-debug",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)

async def upload_hand_landmarks_debug(file: UploadFile = File(...)):
    """
    Debug endpoint for MediaPipe hand landmarks.

    Steps:
      1. Upload image
      2. Detect A4 paper
      3. Warp A4 into a normalized coordinate system
      4. Run MediaPipe hand landmark detection on the warped A4
      5. Draw landmarks and rough measurement lines
      6. Return debug image directly in Swagger
    """

    # Allow only common image formats
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    # Decode uploaded image into OpenCV Bgr format
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image"
        )

    # Detect A4 paper first because it is the reference object
    a4_result = detect_a4(image)

    if not a4_result.get("a4_found") or "corners_px" not in a4_result:
        raise HTTPException(
            status_code=422,
            detail="A4 paper not found, cannot run hand landmark detection"
        )

    # Detect A4 orientation so scale metadata matches portrait/landscape output
    orientation = get_a4_orientation(a4_result["corners_px"])

    # Warp the detected A4 into a normalized coordinate system
    warped, matrix = warp_a4(image, a4_result["corners_px"])

    # Get pixel-to-millimeter scale from the warped A4 size
    scale = get_scale_px_per_mm(orientation)

    try:
        # Run MediaPipe hand landmark detection on the warped A4 image
        hand_result = detect_hand_landmarks(warped)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MediaPipe hand landmark detection failed: {str(e)}"
        )

    # Calculate rough measurement estimates from landmarks
    measurement_result = estimate_hand_measurements(hand_result, scale)

    # Draw debug overlay
    debug_image = draw_hand_landmarks_debug(
        warped,
        hand_result,
        measurement_result=measurement_result
    )

    # Encode debug image as JPEG for Swagger/browser display
    jpeg_bytes = encode_image_jpeg(debug_image)

    headers = {
        "Content-Disposition": "inline; filename=hand_landmarks_debug.jpg",
        "X-A4-Found": "True",
        "X-A4-Orientation": orientation,
        "X-A4-Detection-Method": a4_result.get("detection_method", "unknown"),
        "X-Hand-Found": str(hand_result.get("hand_found", False)),
    }

    if measurement_result:
        headers["X-Approx-Hand-Length-MM"] = str(
            measurement_result["approx_hand_length_mm"]
        )
        headers["X-Approx-Palm-Width-MM"] = str(
            measurement_result["approx_palm_width_mm"]
        )

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers=headers
    )