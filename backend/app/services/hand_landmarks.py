# app/services/hand_landmarks.py
# Handles MediaPipe hand landmark detection on the warped A4 image
# Purpose: detect 21 hand landmark points and create debug overlays for measurement development

from pathlib import Path
import math

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Path to the MediaPipe Hand Landmarker model file
# The model is stored inside the project, not inside the virtual environment
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


# MediaPipe detects 21 hand landmarks
# These names make the output easier to understand and debug
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]


# Connections between landmarks
# These are used to draw a simple hand skeleton on the debug image
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),

    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),

    # Palm connections
    (5, 9), (9, 13), (13, 17),
]


# Cached MediaPipe landmarker instance
# Creating the model every request would be slower
# This keeps one landmarker instance in memory after the first use
_LANDMARKER = None


def get_hand_landmarker():
    """
    Creates or returns the cached MediaPipe Hand Landmarker.

    Returns:
      MediaPipe HandLandmarker instance
    """

    global _LANDMARKER

    if _LANDMARKER is not None:
        return _LANDMARKER

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model file not found: {MODEL_PATH}"
        )

    # BaseOptions tells MediaPipe where the trained .task model file is located
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    # HandLandmarkerOptions controls how the hand model runs
    # running_mode=IMAGE is used because our endpoint processes one uploaded image at a time
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,

        # Detect at most one hand for the first prototype
        # This keeps measurement logic simpler at the beginning
        num_hands=1,

        # Minimum confidence for detecting a hand
        # 0.5 is MediaPipe's common balanced default starting point
        min_hand_detection_confidence=0.5,

        # Minimum confidence that a detected hand is really present
        # 0.5 is a balanced starting point for debug/prototype use
        min_hand_presence_confidence=0.5,

        # Tracking confidence mainly matters more for video/live modes
        # It is kept at 0.5 for consistency with MediaPipe defaults
        min_tracking_confidence=0.5,
    )

    _LANDMARKER = vision.HandLandmarker.create_from_options(options)
    return _LANDMARKER


def detect_hand_landmarks(warped_a4_image):
    """
    Runs MediaPipe hand landmark detection on the warped A4 image.

    Input:
      warped_a4_image: OpenCV BGR image

    Returns:
      dictionary with hand_found, landmarks, handedness, and debug metadata
    """

    image_height, image_width = warped_a4_image.shape[:2]

    # MediaPipe expects RGB image data
    # OpenCV uses BGR by default, so conversion is required
    rgb_image = cv2.cvtColor(warped_a4_image, cv2.COLOR_BGR2RGB)

    # Ensure memory layout is safe for MediaPipe
    rgb_image = np.ascontiguousarray(rgb_image)

    # Create MediaPipe image wrapper
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    landmarker = get_hand_landmarker()
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return {
            "hand_found": False,
            "reason": "No hand landmarks detected",
            "hands": []
        }

    hands = []

    for hand_index, landmarks in enumerate(result.hand_landmarks):
        landmarks_px = []
        landmarks_normalized = []

        for landmark_index, landmark in enumerate(landmarks):
            # MediaPipe returns x/y as normalized values between 0.0 and 1.0
            # Convert them into pixel coordinates on the warped A4 image
            x_px = int(round(landmark.x * (image_width - 1)))
            y_px = int(round(landmark.y * (image_height - 1)))

            # Clamp coordinates inside the image
            # This prevents drawing errors if MediaPipe returns a point slightly outside the image
            x_px = max(0, min(image_width - 1, x_px))
            y_px = max(0, min(image_height - 1, y_px))

            landmarks_px.append({
                "index": landmark_index,
                "name": LANDMARK_NAMES[landmark_index],
                "x": x_px,
                "y": y_px
            })

            landmarks_normalized.append({
                "index": landmark_index,
                "name": LANDMARK_NAMES[landmark_index],
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z)
            })

        # Handedness means whether MediaPipe thinks the detected hand is left or right
        handedness_label = "unknown"
        handedness_score = 0.0

        if result.handedness and len(result.handedness) > hand_index:
            if result.handedness[hand_index]:
                category = result.handedness[hand_index][0]
                handedness_label = category.category_name
                handedness_score = float(category.score)

        hands.append({
            "handedness": handedness_label,
            "handedness_score": handedness_score,
            "landmarks_px": landmarks_px,
            "landmarks_normalized": landmarks_normalized
        })

    return {
        "hand_found": True,
        "hands_detected": len(hands),
        "primary_hand": hands[0],
        "hands": hands
    }


def estimate_hand_measurements(hand_result, scale_info):
    """
    Calculates rough hand measurements from MediaPipe landmarks.

    Current prototype measurements:
      hand length: wrist -> middle fingertip
      palm width: index MCP -> pinky MCP

    These are not final medical-grade measurements.
    They are first prototype values for debugging and development.
    """

    if not hand_result.get("hand_found", False):
        return None

    landmarks = hand_result["primary_hand"]["landmarks_px"]

    # Landmark 0 = wrist
    # Landmark 12 = middle fingertip
    wrist = landmarks[0]
    middle_tip = landmarks[12]

    # Landmark 5 = index finger MCP knuckle
    # Landmark 17 = pinky MCP knuckle
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    hand_length_mm = _distance_mm(
        wrist,
        middle_tip,
        scale_info["px_per_mm_x"],
        scale_info["px_per_mm_y"]
    )

    palm_width_mm = _distance_mm(
        index_mcp,
        pinky_mcp,
        scale_info["px_per_mm_x"],
        scale_info["px_per_mm_y"]
    )

    return {
        "approx_hand_length_mm": round(hand_length_mm, 1),
        "approx_palm_width_mm": round(palm_width_mm, 1),
        "length_definition": "wrist landmark 0 to middle fingertip landmark 12",
        "width_definition": "index MCP landmark 5 to pinky MCP landmark 17"
    }


def _distance_mm(point_a, point_b, px_per_mm_x, px_per_mm_y):
    """
    Calculates distance between two pixel points in millimeters.

    X and Y are converted separately to millimeters.
    This keeps the calculation correct even if x/y scale ever differs slightly.
    """

    dx_mm = (point_b["x"] - point_a["x"]) / px_per_mm_x
    dy_mm = (point_b["y"] - point_a["y"]) / px_per_mm_y

    return math.sqrt(dx_mm ** 2 + dy_mm ** 2)


def draw_hand_landmarks_debug(warped_a4_image, hand_result, measurement_result=None):
    """
    Draws MediaPipe hand landmarks on the warped A4 image.

    Draws:
      hand skeleton
      landmark points
      selected key landmark labels
      debug information bar
    """

    output = warped_a4_image.copy()

    if hand_result.get("hand_found", False):
        landmarks = hand_result["primary_hand"]["landmarks_px"]

        # Draw skeleton connections first
        for start_index, end_index in HAND_CONNECTIONS:
            start = landmarks[start_index]
            end = landmarks[end_index]

            cv2.line(
                output,
                (start["x"], start["y"]),
                (end["x"], end["y"]),
                (0, 200, 0),
                2
            )

        # Draw landmark points
        for point in landmarks:
            cv2.circle(
                output,
                (point["x"], point["y"]),
                5,
                (0, 255, 0),
                -1
            )

        # Label only the most important points to avoid clutter
        key_labels = {
            0: "WRIST",
            5: "INDEX MCP",
            12: "MIDDLE TIP",
            17: "PINKY MCP",
        }

        for index, label in key_labels.items():
            point = landmarks[index]
            cv2.putText(
                output,
                label,
                (point["x"] + 8, point["y"] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

        primary_hand = hand_result["primary_hand"]

        info_lines = [
            "MEDIAPIPE HAND FOUND",
            f"Handedness: {primary_hand.get('handedness', 'unknown')} "
            f"({primary_hand.get('handedness_score', 0):.2f})",
        ]

        if measurement_result:
            info_lines.append(
                f"Approx length: {measurement_result['approx_hand_length_mm']:.1f} mm"
            )
            info_lines.append(
                f"Approx palm width: {measurement_result['approx_palm_width_mm']:.1f} mm"
            )

    else:
        info_lines = [
            "HAND NOT FOUND",
            f"Reason: {hand_result.get('reason', 'unknown')}"
        ]

    _draw_info_bar(output, info_lines)

    return output


def _draw_info_bar(image, lines):
    """
    Draws a semi-transparent information bar at the top of the image.
    """

    bar_h = 28 * (len(lines) + 1)

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1], bar_h),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    y = 22

    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        y += 26