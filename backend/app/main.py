# app/main.py
# Crude for FastAPI APP
# Starts the whole backend-app

# import FastAPI framework
from fastapi import FastAPI 

# import uploadd routes from app/routes/upload.py and rename it here from "router" --> "upload_router"
from app.routes.upload import router as upload_router


# Create FastAPI-app
# Setting title for Swagger UI
app = FastAPI(
    title="Image measurement API"
)

# Adding upload endpoints to the app
app.include_router(upload_router)

# Define GET endpoint "/" (root address)
@app.get("/")

# this function is executed when going into http://127.0.0.1:8000/
def root():

    # return simple confirmation as JSON
    return {"message": "API running"}
 