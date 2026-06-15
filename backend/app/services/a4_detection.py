# app/services/a4_detection.py

# First draft version of A4-detection
# Uses CV
# Edge detection (canny)
# Shape detection (contours)
# geometric filtering (area + aspect ratio) --> this helped to rule out false true detections when testing on images

# !!Notes after testing!!:
# Detects some of the A4 paper image shots, and even some of the ones where A4 is tilted. However it still rejects some of them... 
# Next version should focus on adjusting Canny (might be too sensitive), background noise (different backgrounds)... Ratio logic could also be too fragile.


import cv2


def detect_a4(image):

    # Convert image to grayscale = reduces complexity in detection hence color is not useful for shape detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Blur image for noise reduction
    # GaussianBlur smooth image and removes small noise
    # kernel size set to (5,5)   must be odd numbers for the best results
    # standard deviation 0 = Lets OpenCV automatically calculate the best standard deviation based on the kernel size
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    # Detect edges
    # Parameters set as 50 = lower treshold, 150 = upper treshold
    # when pixel gradient > 150 --> strong edge (kept)
    # between 50-150 -> kept only if connected to strong edge
    # below 50 -> ignored
    # Notes for later on  tuning: Lower values = more edges (but noisy), Higher values = fewer edges (but cleaner)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours = shapes
    # RETR_EXTERNAL retrieval mode to extract only extreme outer boundaries of object, ignoring inner holes
    # CHAIN_APPROX_SIMPLE = compresses points saving memory --> removes unnessecary points
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

     # Sort contours by area (largest first)
     # idea behind this logis = A4 paper is likely the largest object in the image
     # smaller contours are usually noise or texture
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )


    # A4 aspect ratio reference
    # A4 measures: 210x297mm
    # This upgraded my code from shape detector looking for rectangle to --> A4 rectangle detector
    A4_RATIO = 297 / 210

   # best_candidate stores the most likely A4 shape found in the image
   # best_score keeps track of how good each candidate is --> higher score = better match to A4 based on  size and shape similarity wise
    best_candidate = None
    best_score= 0


    # Analyze each contour

    for contour in contours:

        # area = number of pixels inside shape
        area = cv2.contourArea(contour)


        # Filter 1: remove small noise
        # If shape is too small -> ignore it
        if area < 5000:
             continue
        
        

        # minAreaRect = rotated bounding box = rotated A4 paper
        # minAreaRect handles tilted paper
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect


         # avoid division errors
        if w == 0 or h == 0:
            continue

        # aspect ratio of detected shape
        ratio = max(w, h) / min(w, h)

        # how close we are to real A4 shape
        ratio_error = abs(ratio - A4_RATIO)



        # scoring system
        # area = bigger shapes are more likely to be paper
        # ratio_error = penalizes wrong shapes
        # +0.01 avoids division by zero
        score = area / (ratio_error + 0.01)

        # keep best candidate so far
        if score > best_score:
            best_score = score
            best_candidate = {
                "area": area,
                "ratio": ratio,
                "angle": angle
            }

        
     
     # "Final decision" for the best A4 candidate
     # tolerance: 0.4 allows some distortion from perspective
    if best_candidate and abs(best_candidate["ratio"] - A4_RATIO) < 0.4:
        return {
            "a4_found": True,
            "approx_ratio": float(best_candidate["ratio"]),
            "confidence_score": float(best_score),
            "angle": float(best_candidate["angle"])
        }
    
    # no valid A4 found
    return {
        "a4_found": False
    }



        
