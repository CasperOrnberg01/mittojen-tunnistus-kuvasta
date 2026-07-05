# backend/pick_corners.py
# temporary helper script for manually selecting A4 papersheet corners
# run this script on virtual env with an command like this example: python pick_corners.py + image's path

import cv2
import sys


# script expects one command line argument 
if len(sys.argv) < 2:
    print("Usage: python pick_corners.py path/to/image.jpg")
    sys.exit(1)


# read image from path given in terminal
image_path = sys.argv[1]
image = cv2.imread(image_path)


# cv2.imread returns None if the path is wrong or the file cannot be decoded
if image is None:
    print(f"Could not open image: {image_path}")
    sys.exit(1)


# Store original dimensions
# Coordinates sent to the backend must match the original image size, not resized display size
original_h, original_w = image.shape[:2]

# if images are too big for the sreen, display preview is resized
max_display_width = 1200
scale = 1.0

if original_w > max_display_width:
    scale = max_display_width / original_w
    display_w = int(original_w * scale)
    display_h = int(original_h * scale)
    display = cv2.resize(image, (display_w, display_h))
else:
    display = image.copy()


# Points are stored in backend required order:
# top left, top right, bottomright, bottom left
points = []
labels = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
]


# Redraw the image preview with selected corner points and polygon lines
def redraw():
    preview = display.copy()

    # Draw every clicked point and its label
    for i, point in enumerate(points):
        x_display = int(point[0] * scale)
        y_display = int(point[1] * scale)

        cv2.circle(preview, (x_display, y_display), 6, (0, 255, 0), -1)
        cv2.putText(
            preview,
            labels[i],
            (x_display + 8, y_display - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # When all four corners are selected, draw the full A4 quadrilateral
    # If the lines cross, the clicked order is wrong and should be reset
    if len(points) == 4:
        display_points = [
            (int(x * scale), int(y * scale))
            for x, y in points
        ]

        for i in range(4):
            cv2.line(
                preview,
                display_points[i],
                display_points[(i + 1) % 4],
                (0, 255, 0),
                2,
            )

    cv2.imshow("Pick A4 corners", preview)

# Handle mouse clicks in the OpenCV preview window.
#Each left click stores one A4 corner in original image coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        original_x = int(round(x / scale))
        original_y = int(round(y / scale))

        points.append((original_x, original_y))

        print(f"{labels[len(points) - 1]}: x={original_x}, y={original_y}")
        redraw()


print("Click the A4 corners in this order:")
print("1. top_left")
print("2. top_right")
print("3. bottom_right")
print("4. bottom_left")
print("")
print("Press R to reset.")
print("Press ENTER when done.")
print("Press ESC to quit.")


# Create the OpenCV window and connect mouse clicks to mouse_callback()
cv2.namedWindow("Pick A4 corners", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Pick A4 corners", mouse_callback)
redraw()


# Wait for keyboard input
# ENTER finishes selection, R resets selected points, ESC quits
while True:
    key = cv2.waitKey(0)

    if key == 27:
        break

    if key in [13, 10]:
        break

    if key in [ord("r"), ord("R")]:
        points = []
        redraw()
        print("Reset points.")

cv2.destroyAllWindows()


# The backend needs exactly four corners
if len(points) != 4:
    print("You did not select 4 points.")
    sys.exit(1)


# Print values into terminal that will be pasted manually in swagger UI
print("")
print("Paste these into Swagger:")
print("")

for label, point in zip(labels, points):
    print(f"{label}_x = {point[0]}")
    print(f"{label}_y = {point[1]}")


# Also print  poowerShell curl.exe command for quick terminal testing
print("")
print("PowerShell curl.exe example:")
print("")

print('curl.exe -X POST "http://127.0.0.1:8000/upload/hand-landmarks-manual-debug" `')
print('  -H "accept: image/jpeg" `')
print(f'  -F "file=@{image_path};type=image/jpeg" `')

for label, point in zip(labels, points):
    print(f'  -F "{label}_x={point[0]}" `')
    print(f'  -F "{label}_y={point[1]}" `')

print('  --output hand_manual_debug.jpg')