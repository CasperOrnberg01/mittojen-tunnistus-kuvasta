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


# Define GET endpoint for root URL
@app.get("/")

# this function is executed when going into http://127.0.0.1:8000/
def root():

    # return simple confirmation as JSON so we know server is running
    return {"message": "API running"}
 