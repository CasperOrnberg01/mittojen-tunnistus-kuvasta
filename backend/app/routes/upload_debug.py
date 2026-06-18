# app/routes/upload_debug.py
# Visual debug endpoint
# this endpoint returns JPEG image with A4 detection information drawn on top

import cv2 # cv2 is OpenCV, used here to decode uploaded image
import numpy as np # numpy is used to convert raw upload bytes into an array that OpenCV can read

from fastapi import APIRouter, UploadFile, File, HTTPException  # FastAPI route and upload tools
from fastapi.responses import Response  # response lets ur return raw JPEG bytes instead of a json file


from app.services.a4_detection import detect_a4  # detect_a4 finds the A4 paper sheet and returns data: it's corners
from app.services.quality_check import assess_image_quality # asses_image_quality checks blur, brightness and estimated camera tilt angle

# draw_a4_overlay draws the green/yellow/red debug result on the image
# encode_image_jpeg converts the final image into bytes for the browser
from app.services.debug_overlay import draw_a4_overlay, encode_image_jpeg 

# create router for this file
router = APIRouter()

# Create a POST endpoint at /upload/debug
# responses tells Swagger that a successful response is an image/jpe
# response_class=Response tells FastAPI not to automatically convert the result to JSON.
@router.post(
    "/upload/debug",
    responses={200: {"content": {"image/jpeg": {}}}},  # Swagger knows that JPEG
    response_class=Response,
)
async def upload_debug(file: UploadFile = File(...)):
    """
    Debug-endpoint: Upload image →in return you get JPEG-image
    where A4 paper is surrounded with green outlay + image quality measurements up right corner

    Use in swagger:
      1. Open http://127.0.0.1:8000/docs
      2. POST /upload/debug → Try it out → choose image → Execute
      3. Click "Download file" → open image
    """

    # Supported upload formats
    # These match the normal upload endpoint
    ALLOWED = {"image/jpeg", "image/png", "image/webp"}

    # Reject unsupported file formats early
    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported: {file.content_type}"
        )
    
    # Read the uploaded image into memory as bytes
    contents = await file.read()

    # Reject empty files before OpenCV tries to decode them
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")


    # "decoding" image
    # Convert the raw bytes into a NumPy array
    # np.uint8 is the standard byte type expected by cv2.imdecode
    np_array = np.frombuffer(contents, np.uint8)

    # Decode the NumPy byte array into an OpenCV BGR image
    # cv2.IMREAD_COLOR forces a 3-channel color image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # If OpenCV cannot decode the upload, it returns None
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")


    # Check image quality
    # This does not block A4 detection, it only adds debug information and warnings
    quality = assess_image_quality(image)

    # Detect the A4 paper
    # This is always run, even if quality is weak, because debug output is useful
    a4_result = detect_a4(image)


    # Draw the visual overlay on a copy of the image
    # Green means A4 found and quality acceptable
    # Yellow means A4 found but quality weak
    # Red means A4 not found
    annotated = draw_a4_overlay(image, a4_result, quality_result=quality)

    # encode the annotated OpenCV image as JPEG bytes
    jpeg_bytes = encode_image_jpeg(annotated)

    
    # Return the JPEG directly to Swagger/browser
    # Extra headers are useful for curl, frontend debugging, or quick inspection
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            # Browser directly opens the image, doesn't download the file because of "inline"
            "Content-Disposition": "inline; filename=debug_a4.jpg",

            # Add quality info into headers
            "X-A4-Found":      str(a4_result.get("a4_found", False)),
            "X-Quality-OK":    str(quality.get("acceptable", False)),
            "X-Quality-Warns": "; ".join(quality.get("warnings", [])) or "none",
        }
    )
