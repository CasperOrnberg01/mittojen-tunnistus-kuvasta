# app/services/a4_warp.py
# Handles perspective correction for detected A4 paper
# Purpose: convert photographed A4 into a clean, flat, standardsize image

import cv2
import numpy as np


# Real A4 paper width in millimeters
# A4 is 210mm wide in portrait orientation
A4_WIDTH_MM = 210.0

# Real A4 paper height in millimeters
# A4 is 297mm tall in portrait orientation
A4_HEIGHT_MM = 297.0

# Output width of the warped A4 image in pixels
# 794px is approximately 210mm at 96 DPI
# Calculation: 210mm / 25.4mm per inch * 96 pixels per inch = about 794px
A4_WIDTH_PX = 794

# Output height of the warped A4 image in pixels
# 1123px is approximately 297mm at 96 DPI
# Calculation: 297mm / 25.4mm per inch * 96 pixels per inch = about 1123px
A4_HEIGHT_PX = 1123


def warp_a4(image, corners):
    """
    Perspective-corrects the detected A4 paper.

    corners must be in this order:
    [top-left, top-right, bottom-right, bottom-left]

    Returns:
      straightened A4 image
      perspective matrix
    """


    # Destination points describe the final perfect A4 rectangle
    # Top-left becomes (0, 0)
    # Top-right becomes (A4_WIDTH_PX - 1, 0)
    # Bottom-right becomes (A4_WIDTH_PX - 1, A4_HEIGHT_PX - 1)
    # Bottom-left becomes (0, A4_HEIGHT_PX - 1)
    # -1 is used because pixel coordinates start at 0
    dst = np.array([
        [0, 0],
        [A4_WIDTH_PX - 1, 0],
        [A4_WIDTH_PX - 1, A4_HEIGHT_PX - 1],
        [0, A4_HEIGHT_PX - 1]
    ], dtype="float32")

    # Source points are the detected A4 corners from the original photo
    # float32 is required by cv2.getPerspectiveTransform
    src = np.array(corners, dtype="float32")

    # Calculate the 3x3 perspective transform matrix
    # This matrix maps the original A4 corner points into the destination rectangle
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform
    # The output size is fixed to A4_WIDTH_PX x A4_HEIGHT_PX
    warped = cv2.warpPerspective(image, matrix, (A4_WIDTH_PX, A4_HEIGHT_PX))
    
    # Return both the image and the matrix for future measurement steps
    return warped, matrix


def get_scale_px_per_mm():
    """
    Calculates pixels per millimeter after warping.

    Since warped A4 has fixed size:
      width: 794 px = 210 mm
      height: 1123 px = 297 mm
    """
    
    # Horizontal pixels per millimeter
    px_per_mm_x = A4_WIDTH_PX / A4_WIDTH_MM

    # Vertical pixels per millimeter
    px_per_mm_y = A4_HEIGHT_PX / A4_HEIGHT_MM

    # Return both axis-specific values and an average
    # The average is convenient for simple measurements
    return {
        "px_per_mm_x": round(px_per_mm_x, 4),
        "px_per_mm_y": round(px_per_mm_y, 4),
        "px_per_mm_avg": round((px_per_mm_x + px_per_mm_y) / 2, 4),
        "a4_width_mm": A4_WIDTH_MM,
        "a4_height_mm": A4_HEIGHT_MM,
        "warped_width_px": A4_WIDTH_PX,
        "warped_height_px": A4_HEIGHT_PX,
    }