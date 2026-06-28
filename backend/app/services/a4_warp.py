# app/services/a4_warp.py
# Handles perspective correction for detected A4 paper
# Purpose: convert photographed A4 into a clean, flat, standardsize image
# Old version contained flaws, that were spotted in our last group meeting
# Helper code and added changes are commented, starting with "Warp fix"


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


# Warp fix: Added fix to measure distance between two cornr points
# This is used to compare the detected A4's top/bottom side length
# against its left/right side length.

def _distance(p1, p2):
    """
    Calculates distance between two 2D points
    Example:
      p1 = [x1, y1]
      p2 = [x2, y2]
    """

    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# Warp fix: Detect whether A4 is on portrait or landscape
# The A4 detector returns corners in this order:
# [top-left, top-right, bottom-right, bottom-left]
#
# If the top/bottom side is longer than the left/right side, the A4 is landscape in the original image,
# Otherwise it is portrait
def get_a4_orientation(corners):
    """
    Determines whether detected A4 is portrait or landscape

    Returns:
      "portrait" or "landscape"
    """

    pts = np.array(corners, dtype="float32")

    tl, tr, br, bl = pts

    top_width = _distance(tl, tr)
    bottom_width = _distance(bl, br)

    left_height = _distance(tl, bl)
    right_height = _distance(tr, br)

    average_width = (top_width + bottom_width) / 2
    average_height = (left_height + right_height) / 2

    if average_width > average_height:
        return "landscape"

    return "portrait"


def warp_a4(image, corners):
    """
    Perspective-corrects the detected A4 paper.

    corners must be in this order:
    [top-left, top-right, bottom-right, bottom-left]

    Returns:
      straightened A4 image
      perspective matrix
    """

    # Warp Fix: Choose output orientation dynamically
    # Previously every A4 was forced into portrait size: 794 x 1123
    # That distorted and stretched landscape A4 images
    #
    # Now:
    #   portrait  ---> 794 x 1123
    #   landscape --> 1123 x 794
    orientation = get_a4_orientation(corners)

    if orientation == "landscape":
        output_width_px = A4_HEIGHT_PX
        output_height_px = A4_WIDTH_PX
    else:
        output_width_px = A4_WIDTH_PX
        output_height_px = A4_HEIGHT_PX

    # Destination points describe the final perfect A4 rectangle
    # Top left becomes (0, 0)
    # Top right becomes (output_width_px - 1, 0)
    # Bottom right becomes (output_width_px - 1, output_height_px - 1)
    # Bottom left becomes (0, output_height_px - 1)
    dst = np.array([
        [0, 0],
        [output_width_px - 1, 0],
        [output_width_px - 1, output_height_px - 1],
        [0, output_height_px - 1]
    ], dtype="float32")

    # Source points are the detected A4 corners from the original photo
    # float32 is required by cv2.getPerspectiveTransform
    src = np.array(corners, dtype="float32")

    # Calculate the 3x3 perspective transform matrix
    # This matrix maps the original A4 corner points into the destination rectangle
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform
    # The output size now depends on detected A4 orientation
    warped = cv2.warpPerspective(image, matrix, (output_width_px, output_height_px))

    # Return both the image and the matrix for future measurement steps
    return warped, matrix


def get_scale_px_per_mm(orientation="portrait"):
    """
    Calculates pixels per millimeter after warping.

    Since warped A4 has fixed size:
      portrait:  794 px = 210 mm, 1123 px = 297 mm
      landscape: 1123 px = 297 mm, 794 px = 210 mm
    """

    # Warp Fix: Match scale data to warp orientation
    # This keeps x-axis and y-axis measurement logic correct
    if orientation == "landscape":
        output_width_px = A4_HEIGHT_PX
        output_height_px = A4_WIDTH_PX
        real_width_mm = A4_HEIGHT_MM
        real_height_mm = A4_WIDTH_MM
    else:
        output_width_px = A4_WIDTH_PX
        output_height_px = A4_HEIGHT_PX
        real_width_mm = A4_WIDTH_MM
        real_height_mm = A4_HEIGHT_MM

    # Horizontal pixels per millimeter
    px_per_mm_x = output_width_px / real_width_mm

    # Vertical pixels per millimeter
    px_per_mm_y = output_height_px / real_height_mm

    # Return both axis specific values and an average
    # The average is convenient for simple measurements
    return {
        "orientation": orientation,
        "px_per_mm_x": round(px_per_mm_x, 4),
        "px_per_mm_y": round(px_per_mm_y, 4),
        "px_per_mm_avg": round((px_per_mm_x + px_per_mm_y) / 2, 4),
        "a4_width_mm": real_width_mm,
        "a4_height_mm": real_height_mm,
        "warped_width_px": output_width_px,
        "warped_height_px": output_height_px,
    }