# app/main.py
# Crude for FastAPI APP
# Starts the whole backend-app

# import FastAPI framework
from fastapi import FastAPI 

# import uploadd routes from app/routes/upload.py and rename it here from "router" --> "upload_router"
from app.routes.upload import router as upload_router

# import debug upload route.
# This route returns original image with A4 detection information drawn on it
from app.routes.upload_debug import router as debug_router

# import warp debug route
# This endpoint returns the straightened A4 image
from app.routes.upload_warp_debug import router as warp_debug_router

# import compare debug route !THIS CHANGE MADE 25.6.26!
# This endpoint returns original detection view and warped A4 view side by side
from app.routes.upload_compare_debug import router as compare_debug_router

#App hand segment testing!!
# Import MediaPipe hand landmark debug route
# This endpoint returns the warped A4 image with detected hand landmarks drawn on top
from app.routes.upload_hand_landmarks_debug import router as hand_landmarks_debug_router

from app.routes.upload_hand_landmarks_manual_debug import router as hand_landmarks_manual_debug_router
#App hand segment testing!! ends

# Create FastAPI-app
# Setting title for Swagger UI
app = FastAPI(
    title="Image measurement API"
)

# Adding upload endpoints to the app
app.include_router(upload_router)

# register visual debug endpoint, this makes POST /upload available in swagger
app.include_router(debug_router)

# added A4 debug endpoint, this makes POST /upload/debug avaoilable in swagger
app.include_router(warp_debug_router)

#!THIS CHANGE MADE 25.6.26!
# register compare debug endpoint, this makes POST /upload/compare-debug available in swagger
app.include_router(compare_debug_router)


#App hand segment testing!!
# Register MediaPipe hand landmark debug endpoint
# Makes POST /upload/hand-landmarks-debug available in Swagger
app.include_router(hand_landmarks_debug_router)

app.include_router(hand_landmarks_manual_debug_router)
#App hand segment testing!! ends


# Define GET endpoint for root URL
@app.get("/")

# this function is executed when going into http://127.0.0.1:8000/
def root():

    # return simple confirmation as JSON so we know server is running
    return {"message": "API running"}
 