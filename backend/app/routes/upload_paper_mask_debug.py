# app/routes/upload_paper_mask_debug.py
# Temporary debug endpoint for inspecting the papermask fallback path

import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from app.services.a4_detection import (
    detect_a4,
    _create_adaptive_paper_mask,
    _clean_paper_mask,
)
from app.services.debug_overlay import draw_a4_overlay, encode_image_jpeg
from app.services.quality_check import assess_image_quality


router = APIRouter()


@router.post(
    "/upload/paper-mask-debug",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def upload_paper_mask_debug(file: UploadFile = File(...)):
    """
    Temporary debug endpoint.

    Returns a 2x2 JPEG grid:
      1. Original uploaded image
      2. Raw adaptive paper mask
      3. Cleaned paper mask after morphology
      4. Final A4 overlay result

    """

    # Supported upload formats match the other upload endpoints
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported: {file.content_type}"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    # Build the exact masks used by the current papermask fallback
    raw_mask = _create_adaptive_paper_mask(image)
    cleaned_mask = _clean_paper_mask(raw_mask, image.shape[:2])

    # Run normal detection too, so the final panel shows the actual current result
    quality = assess_image_quality(image)
    a4_result = detect_a4(image)
    overlay = draw_a4_overlay(image, a4_result, quality_result=quality)

    # Convert masks to BGR so they can be placed in the same debug grid as photos
    raw_mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    cleaned_mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    # Add labels before resizing so each panel is easy to understand in Swagger
    original_panel = _label_panel(image.copy(), "1 original image")
    raw_panel = _label_panel(raw_mask_bgr, "2 raw paper mask")
    cleaned_panel = _label_panel(cleaned_mask_bgr, "3 cleaned paper mask")
    overlay_panel = _label_panel(overlay.copy(), "4 final A4 overlay")

    debug_grid = _make_debug_grid(
        original_panel,
        raw_panel,
        cleaned_panel,
        overlay_panel,
        panel_width=640,
    )

    jpeg_bytes = encode_image_jpeg(debug_grid)

    raw_ratio = cv2.countNonZero(raw_mask) / float(raw_mask.size)
    cleaned_ratio = cv2.countNonZero(cleaned_mask) / float(cleaned_mask.size)

    response = Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "inline; filename=paper_mask_debug.jpg",
            "X-A4-Found": str(a4_result.get("a4_found", False)),
            "X-A4-Detection-Method": str(a4_result.get("detection_method", "none")),
            "X-Raw-Paper-Mask-Ratio": f"{raw_ratio:.4f}",
            "X-Cleaned-Paper-Mask-Ratio": f"{cleaned_ratio:.4f}",
        },
    )

    # Temporary endpoint: expose headers directly for browser debugging
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"

    return response


def _label_panel(image, label):
    """Draw a small label bar at the top of a debug panel"""

    output = image.copy()
    h, w = output.shape[:2]

    bar_h = max(34, int(h * 0.045))
    overlay = output.copy()

    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, output, 0.30, 0, output)

    cv2.putText(
        output,
        label,
        (12, int(bar_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, min(w, h) / 1200.0),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def _resize_panel(image, width):
    """Resize a panel to a fixed width while keeping aspect ratio"""

    h, w = image.shape[:2]

    if w == 0:
        return image

    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def _pad_to_height(image, target_h):
    """Pad a panel at the bottom so panels in the same row have equal height"""

    h, w = image.shape[:2]

    if h >= target_h:
        return image

    pad = np.zeros((target_h - h, w, 3), dtype=np.uint8)
    return np.vstack([image, pad])


def _make_debug_grid(original, raw_mask, cleaned_mask, overlay, panel_width=640):
    """Create a 2x2 debug grid from four BGR images"""

    panels = [
        _resize_panel(original, panel_width),
        _resize_panel(raw_mask, panel_width),
        _resize_panel(cleaned_mask, panel_width),
        _resize_panel(overlay, panel_width),
    ]

    top_h = max(panels[0].shape[0], panels[1].shape[0])
    bottom_h = max(panels[2].shape[0], panels[3].shape[0])

    top_row = np.hstack([
        _pad_to_height(panels[0], top_h),
        _pad_to_height(panels[1], top_h),
    ])

    bottom_row = np.hstack([
        _pad_to_height(panels[2], bottom_h),
        _pad_to_height(panels[3], bottom_h),
    ])

    return np.vstack([top_row, bottom_row])