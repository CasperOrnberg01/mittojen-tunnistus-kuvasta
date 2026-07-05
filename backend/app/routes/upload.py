# app/routes/upload.py

# Upload route for images endpoint
# This endpoint returns JSON data, no visual debug image --> this was the first draft aka skeleton


# APIRouter groups related endpoints into a separate route file
# UploadFile = FastAPI way to handle right format files
# File tells FastAPI that the file is required
# HTTPException lets us return proper HTTP error responses
from fastapi import APIRouter, UploadFile, File, HTTPException

# process_image contains the actual image decoding and A4 detection pipeline
from app.services.image_processing import process_image


# Create a router for this file
# main.py imports and registers this router with the FastAPI app
router = APIRouter()

# Create a POST endpoint at /upload
# The function below runs when the user uploads an image to this endpoint
@router.post("/upload")

# async for function --> asynchronous function can wait in the background --> allowing rest of the code to keep running
# def upload_image is executed when endpoint is called
# file: UploadFile = variable where FastAPI puts user loaded file (image), User sends image --> UploadFile object ---> file (to this variable)
# = File(...)tells that this comes from HTTP request's file field
async def upload_image(file: UploadFile = File(...)):


    # Only allow image formats OpenCV can reasonably decode in this project
    # JPEG and PNG are common phone/web formats
    # WEBP is included just in case because some browsers and phones may produce it
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: jpeg, png, webp"
        )
    
    # Read the uploaded file bytes asynchronously
    contents = await file.read()


    # Empty uploads cannot be decoded as images
    # 400 means the request itself is invalid
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # Run the image processing pipeline
    # The tryexcept prevents unexpected backend crashes from becoming unclear errors
    try:
        result = process_image(contents)
    except Exception as e:
        # Print the technical error to the server console for debugging
        print(f"DEBUG - process_image error: {e}")

        # Return a clear API error to the client
        # 500 means the server failed while processing a valid request
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )

    # if process_image returns error without raising an exception
    # 422 means the uploaded file was understood, but could not be processed as expected
    if "error" in result:
        raise HTTPException(
            status_code=422,
            detail=result["error"]
        )
    
    # Return the successful JSON result to the caller
    return result
