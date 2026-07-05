# app/routes/upload_hand_landmarks_manual_debug.py
# Manual debug endpoint for MediaPipe hand landmark detection
# Upload image + manually selected A4 corners -> warp A4 -> run MediaPipe -> return landmarks debug image

import math

import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from app.services.a4_warp import warp_a4, get_a4_orientation, get_scale_px_per_mm
from app.services.debug_overlay import encode_image_jpeg
from app.services.hand_landmarks import (
    detect_hand_landmarks,
    draw_hand_landmarks_debug,
    estimate_hand_measurements,
)


router = APIRouter()


@router.post(
    "/upload/hand-landmarks-manual-debug",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def upload_hand_landmarks_manual_debug(
    file: UploadFile = File(...),
    top_left_x: float = Form(...),
    top_left_y: float = Form(...),
    top_right_x: float = Form(...),
    top_right_y: float = Form(...),
    bottom_right_x: float = Form(...),
    bottom_right_y: float = Form(...),
    bottom_left_x: float = Form(...),
    bottom_left_y: float = Form(...),
):
    """
    Manual debug endpoint for MediaPipe hand landmarks.

    This endpoint is used when automatic A4 detection fails or gives weak corners.

    Steps:
      1. Upload image
      2. Receive manually selected A4 corner coordinates
      3. Warp A4 into a normalized coordinate system
      4. Run MediaPipe hand landmark detection on the warped A4
      5. Draw landmarks and rough measurements
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

    # Decode uploaded image into OpenCV BGR format
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image"
        )

    # Manual A4 corners must be sent in this order:
    # top-left, top-right, bottom-right, bottom-left
    manual_corners = [
        [top_left_x, top_left_y],
        [top_right_x, top_right_y],
        [bottom_right_x, bottom_right_y],
        [bottom_left_x, bottom_left_y],
    ]

    _validate_manual_corners(manual_corners, image.shape)

    # Detect A4 orientation so scale metadata matches portrait/landscape output
    orientation = get_a4_orientation(manual_corners)

    # Warp the A4 using manually selected corners instead of automatic detection
    warped, matrix = warp_a4(image, manual_corners)

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

    # Draw debug overlay on the warped A4 image
    debug_image = draw_hand_landmarks_debug(
        warped,
        hand_result,
        measurement_result=measurement_result
    )

    # Encode debug image as JPEG for Swagger/browser display
    jpeg_bytes = encode_image_jpeg(debug_image)

    headers = {
        "Content-Disposition": "inline; filename=hand_landmarks_manual_debug.jpg",
        "X-A4-Input-Mode": "manual_corners",
        "X-A4-Orientation": orientation,
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


def _validate_manual_corners(corners, image_shape):
    """
    Validates manually selected A4 corner coordinates.

    This prevents obvious bad inputs before perspective warping:
      - coordinates must be finite numbers
      - coordinates must be inside the image
      - selected area must be large enough
      - selected shape should roughly match A4 aspect ratio
    """

    image_height, image_width = image_shape[:2]

    for point in corners:
        x, y = point

        if not math.isfinite(x) or not math.isfinite(y):
            raise HTTPException(
                status_code=422,
                detail="Manual corner coordinates must be finite numbers"
            )

        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            raise HTTPException(
                status_code=422,
                detail="Manual corner coordinates must be inside the image"
            )

    pts = np.array(corners, dtype="float32")

    # Check that the selected polygon area is not too small
    selected_area = abs(cv2.contourArea(pts))
    image_area = image_width * image_height
    area_ratio = selected_area / image_area

    if area_ratio < 0.05:
        raise HTTPException(
            status_code=422,
            detail="Manual A4 corner area is too small"
        )

    # Check that the selected quadrilateral roughly has A4 proportions
    tl, tr, br, bl = pts

    top_width = _distance(tl, tr)
    bottom_width = _distance(bl, br)
    left_height = _distance(tl, bl)
    right_height = _distance(tr, br)

    average_width = (top_width + bottom_width) / 2
    average_height = (left_height + right_height) / 2

    if average_width <= 0 or average_height <= 0:
        raise HTTPException(
            status_code=422,
            detail="Manual A4 corners create an invalid shape"
        )

    ratio = max(average_width, average_height) / min(average_width, average_height)
    a4_ratio = 297.0 / 210.0

    # Manual input can be imperfect, so this tolerance is intentionally wider
    # then the automatic detector tolerance
    if abs(ratio - a4_ratio) > 0.45:
        raise HTTPException(
            status_code=422,
            detail="Manual A4 corners do not roughly match A4 aspect ratio"
        )


def _distance(point_a, point_b):
    """
    Calculates Euclidean distance between two 2D points.
    """

    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))
