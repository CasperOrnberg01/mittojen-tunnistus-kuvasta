# app/services/image_processing.py
# Process the picture with OpenCV
# image processing "logic" here

# OpenCV library import
import cv2

# numpy import, to handle images as a matrix
import numpy as np

# import detect_a4 function from a4_detection.py
from app.services.a4_detection import detect_a4


# Function receives image's raw data
def process_image(image_bytes):

    # np.frombuffer turns raw data (bytes) --> into numpy array of uint8 values
    #image_bytes = raw data
    # np.uint8 = image pixel standard size --> 8 bit unsigned integer --> integer is saved as 8 bit (0-255 integers)
    np_array = np.frombuffer(image_bytes, np.uint8)

    # cv2.imdecode = recognition of the image file (JPEG,PNG) --> decodes compressed image bytes ---> into pixel matrix (Ready for OpenCV)
    # Briefly, cv.imdecode turns image file's data (integers made from raw data) --> back into real pixel image that OpenCV can handle
    # IMREAD_COLOR = tells OpenCV to download it as a color image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)




    # Debug: check upload
    # Check's that image is numpy array
    print ("DEBUG - image type:", type(image))


    # Check that image was uploaded succesfully
    if image is None:
        #If OpenCV couldn't read the image

        #Debug message into terminal
        print("DEBUG - image is None")

        # return error
        return {"error": "Invalid image or unsupported format"}
    
    


    # Check that image is not empty
    if image.size == 0:
        # If image is empty

        #return error
        return {"error": "Empty image"}
    



    # Define height and width
    height, width = image.shape[:2]


    # Run A4 detection module
    # Checks if image contains something that looks like A4 paper
    a4_result = detect_a4(image)


    #return image measurements to API
    return {
        "height": height,
        "width": width,
        "a4_detection": a4_result
    }
