# app/routes/upload.py
# Upload route for images endpoint
# upload.py handles the API side (HTTP requests)

# APIRouter = splits API into parts (mini API)
# UploadFile = FastAPI way to handle right format files
# File = Tells this is filefield
from fastapi import APIRouter, UploadFile, File

# importing function process_image from app/services/image_pricessing.py
from app.services.image_processing import process_image


# Creating routes (router) where endpoints are added
router = APIRouter()


# Defining POST end point /upload
# Activates when user sends picture
@router.post("/upload")


# async for function --> asynchronous function can wait in the background --> allowing rest of the code to keep running
# def upload_image is executed when endpoint is called
# file: UploadFile = variable where FastAPI puts user loaded file (image), User sends image --> UploadFile object ---> file (to this variable)
# = File(...)tells that this comes from HTTP request's file field
async def upload_image(file: UploadFile = File(...)):

    # read image data into bits
    contents = await file.read()

    # sending picture for OpenCV
    result = process_image(contents)

    # returning result into the frontend (swagger)
    return result
