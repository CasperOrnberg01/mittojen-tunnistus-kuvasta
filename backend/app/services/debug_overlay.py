# app/services/debug_overlay.py
# Draws visual debug information on top of uploaded images
# This is used by /upload/debug to show what the detector found

import cv2
import numpy as np



def draw_a4_overlay(image, a4_result, quality_result=None):
    """
    Displays recognition result on top of original image

    Colour codes:
      Green  = A4 Found, quality ok
      Yellow = A4 Found, but weak quality
      Red = A4 not Found
    """

    # Work on a copy so the original image is not modified
    output = image.copy()

    # Get image height and width
    # shape is (height, width, channels) for a color image
    h, w = output.shape[:2]


    # If quality_result is missing, assume quality is acceptable
    # Otherwise read the "acceptable" boolean from the quality result
    quality_ok = quality_result is None or quality_result.get("acceptable", True)

    # Read whether A4 detection succeeded
    # Default is False if the key does not exist
    a4_found   = a4_result.get("a4_found", False)


    # Draw the A4 outline only if A4 was found and corner points are available
    if a4_found and "corners_px" in a4_result:
        # Convert corner points into int32 because OpenCV drawing functions use pixel integers
        corners = np.array(a4_result["corners_px"], dtype=np.int32)

        # Use green when quality is OK
        # Use yellow when A4 was found but the image has quality warnings
        # OpenCV uses BGR color order, so (0, 200, 0) is green
        color = (0, 200, 0) if quality_ok else (0, 200, 200)

        # Draw the detected A4 outline
        # isClosed=True connects the last point back to the first point
        # thickness=3 makes the line visible without covering too much of the image
        cv2.polylines(output, [corners], isClosed=True, color=color, thickness=3)


        # Draw a filled circle on each detected corner
        # Radius 8 is large enough to see clearly in debug images
        # -1 means the circle is filled
        for pt in corners:
            cv2.circle(output, tuple(pt), 8, color, -1)


        # Labels show the corner order used by the warp step
        # TL = top-left, TR = top-right, BR = bottom-right, BL = bottom-left
        labels = ["TL", "TR", "BR", "BL"]

        # Pixel offsets keep labels slightly away from the corner circles
        offsets = [(-30, -10), (10, -10), (10, 20), (-30, 20)]

        # Draw each corner label
        for pt, label, offset in zip(corners, labels, offsets):
            # Convert the NumPy point into normal integer pixel coordinates
            pos = (int(pt[0]) + offset[0], int(pt[1]) + offset[1])

            # Draw readable corner text
            # FONT_HERSHEY_SIMPLEX is a built-in OpenCV font
            # 0.55 is the text scale
            # thickness=2 makes the label visible on real photos
            cv2.putText(output, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


        # Text lines shown in the top information bar
        info_lines = [
            f"A4 FOUND",

            # detection_method shows whether the normal contour detector or fallback detector found the A4
            f"Method: {a4_result.get('detection_method', 'contour')}",

            # approx_ratio should be near 1.414 for A4
            f"Ratio: {a4_result.get('approx_ratio', 0):.3f} (target 1.414)",

            # angle is the rotation angle reported by cv2.minAreaRect
            f"Angle: {a4_result.get('angle', 0):.1f} deg",
           
            # area_ratio shows how much of the image the detected candidate covers
            # If this is near 90-100%, the detector may have selected the image border
            f"Area: {a4_result.get('area_ratio', 0) * 100:.1f}% of image",
           
            # confidence_score is a relative internal score
            # It is useful for comparing candidates, not as a perfect probability
            f"Confidence: {a4_result.get('confidence_score', 0):.0f}",
        ]

    else:
        # Draw a red X over the image when A4 was not found
        # OpenCV uses BGR, so (0, 0, 220) is red
        cv2.line(output, (0, 0), (w, h), (0, 0, 220), 3)
        cv2.line(output, (w, 0), (0, h), (0, 0, 220), 3)

        # Show a simple failure message in the info bar
        info_lines = ["A4 NOT FOUND"]


    # Draw the dark transparent information bar
    _draw_info_bar(output, info_lines, quality_result)

    # Return the annotated image
    return output


def _draw_info_bar(image, lines, quality_result=None):
    """Draws transparent info bar on top corner of image"""

    # Each text line needs about 28 pixels of vertical space
    # +1 gives a little extra padding at the bottom
    bar_h = 28 * (len(lines) + 1)

     # If quality information exists, add space for three extra quality lines
    if quality_result:
        bar_h += 28 * 3  # more room for quality

    # Create a copy used only for the dark rectangle overlay
    overlay = image.copy()

     # Draw a dark rectangle across the top of the image
    # (20, 20, 20) is dark gray in BGR
    # -1 means the rectangle is filled
    cv2.rectangle(overlay, (0, 0), (image.shape[1], bar_h), (20, 20, 20), -1)

    # Blend the dark overlay with the original image
    # 0.65 means the dark overlay is stronger than the original background
    # 0.35 keeps some original image visible through the bar
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    
    # Starting y-position for the first text line
    y = 22

    # Draw the main info lines
    for line in lines:
        cv2.putText(image, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 26

    # Draw quality details if they were provided
    if quality_result:
        # Separator line between A4 detection info and quality info
        cv2.line(image, (8, y), (image.shape[1] - 8, y), (80, 80, 80), 1)
        y += 20

        # Build quality text lines
        # Missing values default to 0 or False so the overlay does not crash
        q_lines = [
            f"Blur:  {quality_result.get('blur_score', 0):.0f}  {'OK' if quality_result.get('blur_ok') else 'WEAK'}",
            f"Light: {quality_result.get('brightness', 0):.0f}  {'OK' if quality_result.get('brightness_ok') else 'WEAK'}",
            f"Angle: {quality_result.get('tilt_deg', 0):.1f} deg  {'OK' if quality_result.get('tilt_ok') else 'WEAK'}",
        ]
        # Draw each quality line
        for line in q_lines:
            # Green-ish text for OK values
            # Orange-ish text for weak values
            color = (180, 230, 180) if "OK" in line else (100, 150, 255)
            cv2.putText(image, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 24


def encode_image_jpeg(image, quality=88):
    """Code numpy-image into JPEG-bits."""

    # Encode the image as .jpg
    # cv2.IMWRITE_JPEG_QUALITY controls JPEG compression quality
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # If encoding fails, raise an error so the route can return a clear failure
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")
    
    # Convert the encoded NumPy buffer into normal Python bytes for HTTP response
    return buffer.tobytes()
