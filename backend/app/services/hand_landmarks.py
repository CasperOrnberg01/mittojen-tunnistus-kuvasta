# app/services/hand_landmarks.py
# Handles MediaPipe hand landmark detection on the warped A4 image
# MP detects 21 hand landmarks on warped A4 image, known A4 size gives the px to mm scale, landmark guided segmentation creates binary hand silhoutte... ->
# -> hand length measured from wrist to middle fingertip, palm width measured from real outer contour near mcp kncuckles, landmarks 5 and 17 is used as fallback if contour measurement not trustworthy

from pathlib import Path # path to support model file location
import math # impot normal math fucntions used for distance calculations
import cv2 # OpenCV for masks, contours, drawing and color conversion
import mediapipe as mp # mediapipe's main package for its image wrapper and image format constants
import numpy as np # numpy for coordinate and image array calculations
from mediapipe.tasks import python # mediapipe task base options which are used to point detector to its model file
from mediapipe.tasks.python import vision # mediapipe vision task classes, including hand landmarker and its conf. options

# Path to the MediaPipe Hand Landmarker model file
# The model is stored inside the project, not inside the virtual environment
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


# MediaPipe detects 21 hand landmarks
# Store readable names for mediapipe's 21 landmark indices
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


# Cached MediaPipe landmarker instance, creating model every request would be slower so this keeps one landmarker instance in memory after first use
_LANDMARKER = None


def get_hand_landmarker():
    """create mediapipe detector on first use, then return cached instance"""

    global _LANDMARKER  #tell Python that this function updates module level cached landmarker variable instead of creating a local variable with same name

    if _LANDMARKER is not None: # reuse existing detector after first req, new model creation way slower
        return _LANDMARKER

    if not MODEL_PATH.exists(): # fail with clear message before mp starts if bundled model was mowed or resolved from wrong project path
        raise FileNotFoundError(  
            f"MediaPipe model file not found: {MODEL_PATH}"
        )

    # BaseOptions tells MediaPipe where trained .task model file is located
    base_options = python.BaseOptions(   
        model_asset_path=str(MODEL_PATH)  # give mediapipe filesystem path to hand_landmarker.task model
    )

    # HandLandmarkerOptions controls how the hand model runs
    # running_mode=IMAGE used because endpoint processes one uploaded image at a time
    options = vision.HandLandmarkerOptions(
        base_options=base_options,              # Attach model file settings to hand landmarker conf
        running_mode=vision.RunningMode.IMAGE,  # use single image mode cuz each request processes one uploaded image

        # Only one hand to detect, limited detection to one hand because measurement setup expects one hand on one A4 sheet
        num_hands=1, 

        # Min. confidence for detecting a hand, 0.5 good starting point, require atleast this confidence before mediapipe accepts hand detection
        min_hand_detection_confidence=0.5,

        # Minimum confidence that a detected hand is a hand
        min_hand_presence_confidence=0.5,

        # keep at 0.5 for consistency with MediaPipe defaults, tracking more vital for video modes
        min_tracking_confidence=0.5,
    )

    _LANDMARKER = vision.HandLandmarker.create_from_options(options) #calculate/store cached mediapipe landmarker instance, create configurted mediapipe hand landmarker
    return _LANDMARKER


def detect_hand_landmarks(warped_a4_image):
    """Detect one hand and convert Mediapipe's normalized coordinates into pixels"""

    image_height, image_width = warped_a4_image.shape[:2] # unpack returned values into the image height and width in pixels so each part can be used separately

    # Mediapipe expects RBG image data, OpenCV uses BGR by default, so conversion is required
    rgb_image = cv2.cvtColor(warped_a4_image, cv2.COLOR_BGR2RGB)

    # Ensure memory layout is safe for mediaPipe
    rgb_image = np.ascontiguousarray(rgb_image)  # store the image in one continuos memory block

    # Create mediaPipe image wrapper, calculate this value from current image, landmarks, mask
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,  # tell mediapipe that supplied array contains standard SRGB pixel datas
        data=rgb_image  # give mediapipe the actual RGB image pixels
    )

    # Calculate and store cached hand landmarker
    landmarker = get_hand_landmarker()    
    result = landmarker.detect(mp_image)  # calculate and store result returned, run mediapipe inference and ask model to find hand and 21 landmarks

    # stop before measurement code if mp returned no landmark set
    if not result.hand_landmarks:
        return {  # return structured result dict for debug information
            "hand_found": False,
            "reason": "No hand landmarks detected",
            "hands": []
        }

    hands = [] # model only for one hand, but keeping as list follows mp's normal result struct.
    # convert every detected hand into both pixel coords for opencv and original normalized coordinates for future processing
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        landmarks_px = [] 
        landmarks_normalized = [] 
        # multiply normalized x/y by image size to place ach point on the warped image
        for landmark_index, landmark in enumerate(landmarks): 
            x_px = int(round(landmark.x * (image_width - 1)))  #convert mp normalized x value (0 to 1) unti horizontal px coord. subtracting 1 keeps the max value inside image
            y_px = int(round(landmark.y * (image_height - 1)))  #convert mp normalized y value (0 to 1) unti vertical px coord. subtracting 1 keeps the max value inside image

            # clamp small prediction overshoots so later indexing and drawing cant access pixels outside image
            x_px = max(0, min(image_width - 1, x_px)) 
            y_px = max(0, min(image_height - 1, y_px))
            # pixel coordinates are used for mask creating, drawing and distances
            landmarks_px.append({  # add a new item to list so all measurements are preserved
                "index": landmark_index, 
                "name": LANDMARK_NAMES[landmark_index],
                "x": x_px,  
                "y": y_px
            })

            landmarks_normalized.append({  # add a new item to list so all measurements are preserved 
                "index": landmark_index,  
                "name": LANDMARK_NAMES[landmark_index],  
                "x": float(landmark.x),  
                "y": float(landmark.y),   
                "z": float(landmark.z)    
            }) 

        # Handedness = whether MediaPipe thinks the detected hand is left or right
        handedness_label = "unknown"  # calculate and store detected hand label, left/right
        handedness_score = 0.0        # calculate and store mediapipe's confidence for the handness label

        # read left/right classification only when mp returned entry for this hand
        if result.handedness and len(result.handedness) > hand_index: 
            if result.handedness[hand_index]:
                category = result.handedness[hand_index][0] 
                handedness_label = category.category_name    
                handedness_score = float(category.score)   
        # keep one complete record for this hand: handedness + both coordinate formats
        hands.append({ 
            "handedness": handedness_label,
            "handedness_score": handedness_score,
            "landmarks_px": landmarks_px,
            "landmarks_normalized": landmarks_normalized
        }) 

    return {  # Return structured dictionary describing success or failure result
        "hand_found": True,           
        "hands_detected": len(hands),
        "primary_hand": hands[0],
        "hands": hands
    }


def estimate_hand_measurements(hand_result, scale_info, warped_a4_image=None):
    """
    Calculates hand measurements from MediaPipe landmarks

    Current measurements:
      hand length: wrist -> middle fingertip
      palm width: outer hand contour near the MCP "knuckle" level
    
    Fallback if outer hand contour segmentation fails
    """
    # measurements require valid primary hand,    None = nothing to measure from numeric result of zero
    if not hand_result.get("hand_found", False):
        return None
    # use pixel landmarks, A4 scale converts pixel distances --> mm
    landmarks = hand_result["primary_hand"]["landmarks_px"] 

    # select four landmarks used by current definitions
    wrist = landmarks[0] #Landmark 0 = wrist 
    middle_tip = landmarks[12]#Landmark 12 = middle fingertip

    index_mcp = landmarks[5] # Landmark 5 = index finger MCP knuckle
    pinky_mcp = landmarks[17] # Landmark 17 = pinky MCP knuckle

    # hand length = from wrist to middle finger tip
    hand_length_mm = _distance_mm(wrist, middle_tip, scale_info["px_per_mm_x"], scale_info["px_per_mm_y"])

    # landmark width is kept for comparison and as a safe fallback, MCP landmarks are at joints so this sligthly underestimates real palm width
    landmark_palm_width_mm = _distance_mm(index_mcp, pinky_mcp, scale_info["px_per_mm_x"], scale_info["px_per_mm_y"])

    contour_measurement = None

    # Outer contour measurement needs actual warped image, keeping warped image optional so older callers dont break
    if warped_a4_image is not None:
        contour_measurement = _measure_outer_palm_width(  # Try to measure real outside palm width from the image and landmarks
            warped_a4_image=warped_a4_image,  # pass warped a4 image
            landmarks=landmarks,              # pass 21 detected landmark points for guiding 
            scale_info=scale_info,            # pass x/y pixel per millimeter values calculated from warped A4
            hand_length_mm=hand_length_mm,    # pass already calculated hand length for sanity checks
            landmark_palm_width_mm=landmark_palm_width_mm,  # pass internal mcp width for comparison + fallback
        )

    # prefer real outer edges only when all mask, geometry and consistency checks pass, otherwise use internal landmark width
    if contour_measurement and contour_measurement.get("measurement_ok", False):  
        palm_width_mm = contour_measurement["outer_palm_width_mm"]  # calc./store palm width selected as final result 
        palm_width_method = contour_measurement.get(                # store name of algo that produced final width.
            "palm_width_method",
            "outer_contour_mcp_band"
        )
        width_definition = contour_measurement.get( 
            "width_definition",
            "outer hand contour measured near the MCP knuckles"
        )
    else:  # use this condition when previous condition was not met
        palm_width_mm = landmark_palm_width_mm                               # use internal mcp widtyh only when no trustworthy outer contour width produced
        palm_width_method = "landmark_5_to_17_fallback"                      # keep record of used fallback
        width_definition = "index MCP landmark 5 to pinky MCP landmark 17" 

    result = {
        "approx_hand_length_mm": round(hand_length_mm, 1),
        "approx_palm_width_mm": round(palm_width_mm, 1),
        "landmark_palm_width_mm": round(landmark_palm_width_mm, 1),
        "outer_palm_width_mm": None,
        "palm_width_method": palm_width_method,
        "hand_mask_found": False,
        "length_definition": "wrist landmark 0 to middle fingertip landmark 12",
        "width_definition": width_definition,
    }
    # preserve mask method, contour points, samples and failure reason for debug + headers
    if contour_measurement:
        result.update({
            "outer_palm_width_mm": contour_measurement.get("outer_palm_width_mm"),
            "hand_mask_found": contour_measurement.get("hand_mask_found", False),
            "palm_width_points_px": contour_measurement.get("palm_width_points_px"),
            "palm_width_samples_mm": contour_measurement.get("palm_width_samples_mm", []),
            "hand_contour_px": contour_measurement.get("hand_contour_px", []),
            "segmentation_reason": contour_measurement.get("reason", "unknown"),
            "hand_mask_method": contour_measurement.get("hand_mask_method", "unknown"),
        })

    return result


def _measure_outer_palm_width(warped_a4_image, landmarks, scale_info, hand_length_mm, landmark_palm_width_mm,):
    """Create hand silhoutte and measure palm width from the actual outside hand contour, several checks prevent shadows becoming palm edge"""

    hand_mask_result = _create_landmark_guided_hand_mask(  #Create binary hand silhoutte needed for outer edge measurement
        warped_a4_image,
        landmarks
    )
    # stop if segmentation failed
    if not hand_mask_result.get("hand_mask_found", False): # error handling for debug and to prevent unreliable data from reaching ltr steps
        return {
            "measurement_ok": False,  # whether method produced trustworthy outer width
            "hand_mask_found": False, # whteher usable hand silhoutte was created
            "hand_contour_px": [],    # store outer hand contour for debugging
            "hand_mask_method": hand_mask_result.get("hand_mask_method", "none"),  # store segmentaton method that created handmask
            "reason": hand_mask_result.get("reason", "Hand mask not found")        # explanation of success or failure
        }

    hand_mask = hand_mask_result["mask"]              # calc/store binary image, white pixel= hand, black = background
    landmarks_array = _landmarks_to_array(landmarks)  # calc and store 21 hand points

    # Keep mask information in every result after segmentation succees, contour kept for debugging even if width rejected
    mask_debug = {
        "hand_mask_found": True,
        "hand_contour_px": hand_mask_result.get("hand_contour_px", []),
        "hand_mask_method": hand_mask_result.get("hand_mask_method", "unknown"),
    }

    wrist = landmarks_array[0]       # calc/store landmark 0 at wrist
    index_mcp = landmarks_array[5]   # calc/store landmark 5 at index finger knuckle (mcp)
    middle_mcp = landmarks_array[9]  # calc/store landmark 9 at middle finger knucklle (mcp)
    ring_mcp = landmarks_array[13]   # calc/store landmark 13 at ring finger knuckle (mcp)
    pinky_mcp = landmarks_array[17]  # calc/store landmark 17 at pinky finger knuckle (mcp)

    # The MCP midpoint gives the upper palm (knuckle) level, average the four mcp knuckle points to estiamte center of upper palm
    mcp_midpoint = np.mean(
        np.array([index_mcp, middle_mcp, ring_mcp, pinky_mcp], dtype="float32"),
        axis=0  # array dimension
    )

    palm_axis = mcp_midpoint - wrist  # substract wrist coords from mcp midpoint, result points along hand from wrist towards knuckle and define hand length direction
    palm_axis_length = float(np.linalg.norm(palm_axis))  # calc and store wrist to mcp center distance in pixels

    if palm_axis_length < 10.0: # palm axis = wrist to knuckle center, if under 10 hand is too small or landmarks issues
        return {
            "measurement_ok": False,
            **mask_debug,
            "reason": "Palm axis was too short for contour measurement"
        }

    palm_axis = palm_axis / palm_axis_length  # normalize palm axis length to one

    # Start with the index to pinky MCP direction and remove any component that points along the palm
    width_axis = index_mcp - pinky_mcp  # start sideways palm direction with vector from pinky mcp to index mcp
    width_axis = width_axis - (np.dot(width_axis, palm_axis) * palm_axis)  # remove any part of width vector that points along wrist to finder direction
    width_axis_length = float(np.linalg.norm(width_axis))  #calc and store pixel length of calculated sideways palm before normalization

    # width axis = sideways direction acxross palm after removing any wrist to finger component 
    if width_axis_length < 5.0:  # under 5 pixel it is too close to zero to normalize safely 
        width_axis = np.array([-palm_axis[1], palm_axis[0]], dtype="float32")  # build 90deg meeting point to palm axis, safe fallback
    else:
        width_axis = width_axis / width_axis_length  # normalize sideways palm vector to length 1 to be used as direction

    # Keep width axis pointing from pinky side toward index side, sanity condition
    if np.dot(width_axis, index_mcp - pinky_mcp) < 0:
        width_axis = -width_axis  # reverse sideways vector so it sign consistently points from pinky toward index side

    landmark_span_px = float(np.linalg.norm(index_mcp - pinky_mcp)) # internal mcp span used to scale search distance and reject widths that are way too small or large for this specific hand

    if landmark_span_px < 10.0:  # under 10 pixels means hand is too small or landmarks failed
        return {
            "measurement_ok": False,
            **mask_debug,
            "reason": "Landmark palm width was too short for contour measurement"
        }

    # First try local siderays from index and pinky MCP regions, local rays look outwards from each mcp side separately
    side_ray_measurement = _measure_outer_palm_width_from_side_rays(
        hand_mask=hand_mask,
        index_mcp=index_mcp,
        middle_mcp=middle_mcp,
        ring_mcp=ring_mcp,
        pinky_mcp=pinky_mcp,
        palm_axis=palm_axis,
        palm_axis_length=palm_axis_length,
        landmark_span_px=landmark_span_px,
        scale_info=scale_info,
        hand_length_mm=hand_length_mm,
        landmark_palm_width_mm=landmark_palm_width_mm,
    )
    # return immediately when preferred method produced enough consistent samples
    if side_ray_measurement.get("measurement_ok", False):
        return {
            **side_ray_measurement,
            **mask_debug,
        }

    line_results = [] # create empty list for backup full line measurements that pass all geometry checks

    # test several parallel lines sligthly below knuckles
    for wrist_shift_fraction in (0.08, 0.12, 0.16, 0.20, 0.24, 0.28):  # test several nearby levels moving from knucle row toward wrist
        center = mcp_midpoint - (palm_axis * palm_axis_length * wrist_shift_fraction)  # move candidate line from knucle row towartd wrist

        line_result = _measure_mask_width_on_line(  # calculate and store one candidate
            hand_mask=hand_mask,                    # pass binary hand silhoutte
            center=center,                          # pass expected center
            width_axis=width_axis,                  # pass unit sideways direction across the palm
            landmark_span_px=landmark_span_px,      # pass mcp span in pixels so search distances and width limits scale with this hand
        )
        # this level did not cross one beliavable continous palm segment
        if line_result is None: # skip this candidate because it failed required check, then test next candidate instead
            continue
        # convert landmark to landmark pixel distance into millimetres using A4 scale
        width_mm = _distance_mm_xy(
            line_result["point_a"],
            line_result["point_b"],
            scale_info["px_per_mm_x"],
            scale_info["px_per_mm_y"]
        )

        # A real outside contour should normally be a little wider than internal mcp to mcp landmark distance but not too wide
        # compare outside contour width with internal mcp width, small 0.001 floor prevents division by zero  if earlier value is unexpectedly zero
        width_ratio = width_mm / max(landmark_palm_width_mm, 0.001)
        # outside palm width should be atleast about the internal mcp width, values outside this range potentially means line hit a finger gap, thumb or shadow
        if width_ratio < 0.98 or width_ratio > 1.60:
            continue

        #compare palm width with total hand length as anatomy check.
        hand_ratio = width_mm / max(hand_length_mm, 0.001)
        # check palm width against total hand length. 22% -> 68% range to ereject obvious mistakes
        if hand_ratio < 0.22 or hand_ratio > 0.68: 
            continue
        
        line_result["width_mm"] = float(width_mm)
        line_result["width_ratio"] = float(width_ratio)
        line_results.append(line_result)
    # backup method needs atleast 3 valid parallel measurements, fewer samples = more prone to errors in final width
    if len(line_results) < 3:
        return {
            "measurement_ok": False,
            **mask_debug,
            "palm_width_samples_mm": [
                round(float(item["width_mm"]), 1)
                for item in line_results
            ],
            "side_ray_reason": side_ray_measurement.get("reason", "Side-ray measurement failed"),
            "reason": (
                "Not enough stable outer palm-width lines were found; "
                f"side rays: {side_ray_measurement.get('reason', 'failed')}"
            )
        }
    
    widths = np.array(
        [line_result["width_mm"] for line_result in line_results],
        dtype="float32"
    )
    
    median_width = float(np.median(widths))  # calc and store middle sample width, used because it resists one large or small sample
    median_absolute_deviation = float(np.median(np.abs(widths - median_width)))  # measure typical sample disagreement around median

    # Reject a mask if nearby parallel lines disagree too much: (10%)
    if median_width <= 0 or (median_absolute_deviation / median_width) > 0.10:
        return {
            "measurement_ok": False,
            **mask_debug,
            "palm_width_samples_mm": [round(float(value), 1) for value in widths.tolist()],
            "side_ray_reason": side_ray_measurement.get("reason", "Side-ray measurement failed"),
            "reason": (
                "Outer palm-width samples were not consistent enough; "
                f"side rays: {side_ray_measurement.get('reason', 'failed')}"
            )
        }

    # Use real sampled line closest to the median for the debug overlay
    selected_line = min(
        line_results,
        key=lambda line_result: abs(line_result["width_mm"] - median_width)
    )

    return {
        "measurement_ok": True,
        **mask_debug,
        "outer_palm_width_mm": round(median_width, 1),
        "palm_width_points_px": [
            [int(round(selected_line["point_a"][0])), int(round(selected_line["point_a"][1]))],
            [int(round(selected_line["point_b"][0])), int(round(selected_line["point_b"][1]))],
        ],
        "palm_width_samples_mm": [round(float(value), 1) for value in widths.tolist()],
        "palm_width_method": "outer_contour_mcp_band",
        "width_definition": (
            "outer hand contour measured across several parallel lines "
            "slightly below the MCP knuckles"
        ),
        "reason": "Outer contour width measured successfully with MCP band lines",
    }

def _create_landmark_guided_hand_mask(warped_a4_image, landmarks):
    """build binary handmask guided by landmarks, grabcut and color cleanup"""

    image_height, image_width = warped_a4_image.shape[:2]  # unpack returned values into image heigth and width in pixels so each part can be used separately
    landmark_points = _landmarks_to_array(landmarks)       # calc/store all hand landmarks stored as an x/y np array

    wrist = landmark_points[0]        # Calculate and store landmark 0 at the wrist
    middle_tip = landmark_points[12]  # Calculate and store landmark 12 at the middle fingertip
    index_mcp = landmark_points[5]    # Calculate and store landmark 5 at the index finger knuckle (mcp)
    pinky_mcp = landmark_points[17]   # Calculate and store landmark 17 at the pinky knuckle (mcp)

    hand_length_px = float(np.linalg.norm(middle_tip - wrist))  # Calculate and store the wrist to middlefinger tip distance in pixels
    palm_span_px = float(np.linalg.norm(index_mcp - pinky_mcp)) #  # Calculate and store index MCP to pinky MCP distance in pixels
    # require enough pixels between important landmarks before segmentation, fail before creating a noisy mask
    if hand_length_px < 30.0 or palm_span_px < 12.0:
        return {
            "hand_mask_found": False,
            "hand_mask_method": "none",
            "reason": "Hand landmarks were too small for segmentation"
        }
    # calc and store extra pixels around landmark bounding box so complete hand remains inside the crop
    padding = int(round(max(35.0, hand_length_px * 0.16, palm_span_px * 0.60))) 

    min_x = max(0, int(np.floor(np.min(landmark_points[:, 0]))) - padding)  # calc left edge of the hand crop 
    max_x = min(image_width, int(np.ceil(np.max(landmark_points[:, 0]))) + padding + 1)  # calc right edge of hand crop
    min_y = max(0, int(np.floor(np.min(landmark_points[:, 1]))) - padding)  # calc top edge of hand crop
    max_y = min(image_height, int(np.ceil(np.max(landmark_points[:, 1]))) + padding + 1)  # calc bottom edge of hand crop
    # reject crop smaller than 30 pixels in either direction because grabcut needs greater image region
    if max_x - min_x < 30 or max_y - min_y < 30:  
        return {
            "hand_mask_found": False,  # store whether usable binary hand silhouette was created in returned dict.
            "hand_mask_method": "none",  # store segmentation method that was used tyo create hand mask in the returned dict.
            "reason": "Hand segmentation crop was too small"  # store explanation of result with reason
        }

    crop = warped_a4_image[min_y:max_y, min_x:max_x].copy()  # calc cropped image region containing hand
    local_points = landmark_points - np.array([min_x, min_y], dtype="float32")  # calc landmark coordinates from full image coords into crop coords.

    crop_height, crop_width = crop.shape[:2]  # unpack returned values into hand crop height and width in pxls

    # GrabCut mask values: 0 = sure background   //  1 = sure foreground,  //  2 = probable background,  // 3 = probable foreground
    grabcut_mask = np.full(  # calc grabcut four state mask describing probable foreground and background.
        (crop_height, crop_width), 
        cv2.GC_PR_BGD,
        dtype="uint8"  # uint8 used for numpy data required for this task
    )

    seed_radius = max(3, int(round(palm_span_px * 0.045)))  # calc radius of definite hand seed circles around landmarks
    probable_thickness = max(7, seed_radius * 4)  # calc line thickness used to connect hand regions
    sure_thickness = max(3, seed_radius * 2)  # calc and store sure_thickness for later use

    probable_foreground = np.zeros((crop_height, crop_width), dtype="uint8")  # calc/store pixels liekly to belong to hand before grabcut refines them
    sure_foreground = np.zeros((crop_height, crop_width), dtype="uint8")  # calc/store pixels that are surely known to be belong to hand

    # draw skeleton that connects every landmark chain, continous probable hand region
    for start_index, end_index in HAND_CONNECTIONS:  # loop through start_index, end_index in HAND_CONNECTIONS so same processing is applied to every relevant sample
        start = tuple(np.round(local_points[start_index]).astype("int32"))  # calc/store "start" for later use
        end = tuple(np.round(local_points[end_index]).astype("int32"))  # calc/store "end" for later use
        # draw a line between two pixel points
        cv2.line(  
            probable_foreground,
            start,
            end,
            255,
            probable_thickness
        )
    # add circles around each landmark
    for point in local_points: # calc/store point being moved into hand mask
        point_xy = tuple(np.round(point).astype("int32"))
        # draw a circular landmark endpoint
        cv2.circle(
            probable_foreground,
            point_xy,
            probable_thickness,
            255,
            -1
        )
        # draw a circular landmark endpoint
        cv2.circle(
            sure_foreground,
            point_xy,
            seed_radius,
            255,
            -1
        )

    # Fill the central palm shape as sure foreground, this region contains no  gaps between fingers and is normally reliable
    palm_indices = [0, 1, 2, 5, 9, 13, 17]  # calc/store landmark indices that describe solid palm region
    palm_hull = cv2.convexHull(  # calc/store convex polygon around main palm landmarks and build smallest convex polygon around selected palm landmarks
        np.round(local_points[palm_indices]).astype("int32")
    )
    cv2.fillConvexPoly(sure_foreground, palm_hull, 255)  # fill palm polygon so its insides becomes a likely hand region

    # Add narrower sure foreground skeleton lines
    for start_index, end_index in HAND_CONNECTIONS:
        start = tuple(np.round(local_points[start_index]).astype("int32"))
        end = tuple(np.round(local_points[end_index]).astype("int32"))
        # draw a line between two pixel points
        cv2.line(  
            sure_foreground,
            start,
            end,
            255,
            sure_thickness
        )

    # Mark safe central parts of the gaps between adjacent fingers as sure background, shadows ofthen fill these gaps and make grabcut connect two fingers together
    # this fix was a major step on preventing shadows from "dragging" finger or hand outlines into shadows
    finger_gap_background = _create_finger_gap_background_mask(  # calc/store mask marking spaces between fingers as definite background
        local_points=local_points,  # calc/store landmark coordinates from full image coords into crop coords
        image_shape=(crop_height, crop_width),  # calc/store iamge_shape for later use
        palm_span_px=palm_span_px,  # calc/store index MCP to pinky MCP distance in pixels
    )

    # Expand probable foreground slightly so GrabCut can reach the real skin edges
    dilation_size = max(7, int(round(palm_span_px * 0.12)))  # calc/store template used to expand likely hand region 
    if dilation_size % 2 == 0:  # safety check condition before continuing
        dilation_size += 1  # update template size used to expand likely hand region using new value instead of creating separate variable

    probable_foreground = cv2.dilate(  # calc/store pixels likely to belong to hand before grabcut refines them, expand mask region so nearby pixels are included in search area
        probable_foreground, 
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size, dilation_size)
        ),
        iterations=1  # store iterations for later use
    )

    grabcut_mask[probable_foreground > 0] = cv2.GC_PR_FGD
    grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

    # Explicit finger gap background overrides wider probable foreground skeleton, 
    # This helps the final mask keep every visible gap open even when a dark hand shadow appears between two fingers
    grabcut_mask[finger_gap_background > 0] = cv2.GC_BGD

    # A thin outer crop border is safe background guidance.
    border_size = min(4, max(1, min(crop_height, crop_width) // 40))  # calc border_size
    grabcut_mask[:border_size, :] = cv2.GC_BGD
    grabcut_mask[-border_size:, :] = cv2.GC_BGD
    grabcut_mask[:, :border_size] = cv2.GC_BGD
    grabcut_mask[:, -border_size:] = cv2.GC_BGD

    background_model = np.zeros((1, 65), dtype="float64")  # calc GC background colours
    foreground_model = np.zeros((1, 65), dtype="float64")  # calc GC hand colors

    grabcut_component = None  # store final usable connected component from grabcut
    grabcut_was_shadow_refined = False  # store whether safer shadow cleaned component replaced raw GC component
    # Run opencv operation inside try block to avoid unusual images raising an error
    try:
        cv2.setRNGSeed(12345)  # TEST LINE FOR SAME IMAGE OUTER CONTOUR DET FAIL FIX !!!!!! This idea is to make same image and landmark seeds now start GC from same random state every request
        cv2.grabCut(  # run GC so image colors and landmark seeds can be combined into foreground / background classification
            crop,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            4,
            cv2.GC_INIT_WITH_MASK
        )
        # calc/store binary handmask inside cropped hand image
        crop_mask = np.where(  
            (grabcut_mask == cv2.GC_FGD) | 
            (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype("uint8")

        # Clean tiny isolated pixels and small holes without heavily changing contour
        crop_mask = cv2.morphologyEx(
            crop_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )
        crop_mask = cv2.morphologyEx(  # calc/store binary handmask inside cropped hand image, connect mask pixels with morphology
            crop_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        # Reopen safe finger gaps after morphology, closing operation must not reconnect two fingers through a shadow
        crop_mask[finger_gap_background > 0] = 0
        # calc/store connected hand component before shadow cleanup, select connected foreground object supported by hand landmarks
        raw_grabcut_component = _keep_component_supported_by_landmarks(
            crop_mask,
            local_points
        )

        #learn skin and paper color from this iamge, then remove attached neautral shadows while preserving landmark supported hand pxls
        shadow_refined_mask = _refine_grabcut_mask_against_shadows(  # calc/store GC foreground after removing shadows, remove pxls that look more like shadow than skin
            crop=crop,  # store cropped iamge region containing hand
            crop_mask=crop_mask,  # store binary hand mask inside cropped hand image
            sure_foreground=sure_foreground,  #store pixel confidently known to belong to hand 
            probable_foreground=probable_foreground,  # store pixels likely to belond to hand before GC refines them
            finger_gap_background=finger_gap_background,  # store mask marking spaces between fingers as definite background
            palm_span_px=palm_span_px,  # store index mcp to pinky mcp distance in pixels
        )

        refined_component = None  # store connected hand component after shadow cleanup
        # perform safety condition
        if shadow_refined_mask is not None:  
            refined_component = _keep_component_supported_by_landmarks(  # select connected foreground object supported by hand landmarks
                shadow_refined_mask,
                local_points
            )
        # use refinement only when it keeps enoguh area and landmark coverage, otherwise retain raw GC
        if _is_refined_component_safe(
            raw_component=raw_grabcut_component,  # calc connected hand component before refinement
            refined_component=refined_component,  # calc connected hand component after shadow cleanup
            local_points=local_points,  # calc andf store coordinates from full image coords into crop coords
        ):
            grabcut_component = refined_component  # calc/store final usable connected component from GC
            grabcut_was_shadow_refined = True      # check whether safer shadow cleaned compoenent replaced raw GC componment
        else:  # use alternative path when prev condition was not met
            grabcut_component = raw_grabcut_component  # calc/store final usable connected component produced by GC
            grabcut_was_shadow_refined = False         # check whether safer shadow cleaned component replaced raw GC component
    # handle opencv failure without crashing API request allowing safer fallback segmentation to run
    except cv2.error:
        grabcut_component = None  # calc/store final usable connected component produced by GC
    # perform safety condition
    if grabcut_component is not None:
        selected_component = grabcut_component  # store connected foreground component chosen as hand
        # perform safety condition 
        if grabcut_was_shadow_refined:
            hand_mask_method = "landmark_guided_grabcut_shadow_refined"  # store name of segmentation method that produced selected hand mask
        else:  # use alternative path when prev condition not met
            hand_mask_method = "landmark_guided_grabcut"  # store name of segm. method that produced selected hand mask

    else:  # another alternative when prev cond not met
        # Only use adaptive color segmentation when original GC path did not create usable landmark supported component
        adaptive_mask = _create_adaptive_color_fallback_mask( # calc backup handmask created from adaptive img color models, try image specific color segmentation when GC does not produce usable hand
            crop=crop,  # store cropped image region containing hand 
            probable_foreground=probable_foreground,  # store pixels likely to belong to hand before GC refines them
            sure_foreground=sure_foreground,  # store pixels confidently known to belong to hand
        )

        selected_component = None  # calc/store connected foreground comp chosen as hand
        
        if adaptive_mask is not None:
            selected_component = _keep_component_supported_by_landmarks(  # calc connected foreground comp chosen as hand, select connected fg object supported by hand landmarks
                adaptive_mask,
                local_points
            )
        # fallback with contour measurement if GC and adapt. color fail
        if selected_component is None:
            return {  # return dict result for debugging
                "hand_mask_found": False,
                "hand_mask_method": "none",
                "reason": "No connected hand-mask component matched the palm landmarks"
            }

        hand_mask_method = "adaptive_color_fallback"  # calc name of seg method that produced slected hand mask
    # unpack returned values into all external contours found in binary hand mask, "_" so each part can be used separately
    contours, _ = cv2.findContours(
        selected_component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # selected white component should have outline, none = return segmentation failure
    if not contours:
        return {  # return dict for debugging
            "hand_mask_found": False,
            "hand_mask_method": hand_mask_method,
            "reason": "The selected hand component had no external contour"
        }
    # calc largest hand contour before optional point reduction
    largest_contour_raw = max(contours, key=cv2.contourArea)

    # Width measurement uses the solid external silhouette. GC can leave bg holes inside otherwise correct hand outline, those holes should not split palm width ray
    solid_component = np.zeros_like(selected_component)  # calc filled version of outer contour with internal holes removed
    cv2.drawContours(  # draw selected contour into a mask
        solid_component,
        [largest_contour_raw],  # start grouped expression
        contourIdx=-1,  # calc "contourIdx" for later use 
        color=255,      # calc "color" for later use
        thickness=cv2.FILLED  # calc thickness in pixels of a drawn mask region
    )
    # calc crop sized hand mask placed back into full warped image coords
    full_mask = np.zeros((image_height, image_width), dtype="uint8")
    full_mask[min_y:max_y, min_x:max_x] = solid_component
    # calc largest external contour expected to outline full hand
    largest_contour = largest_contour_raw.reshape(-1, 2)

    # Store a simplified contour for the debug overlay, avoids carrinyg bunch of identical contour pxls
    if len(largest_contour) > 240:
        step = int(math.ceil(len(largest_contour) / 240.0))
        largest_contour = largest_contour[::step]

    largest_contour = largest_contour + np.array([min_x, min_y], dtype="int32")  # calc largest external contour expected to outline the full hand
    hand_contour_px = largest_contour.astype("int32").tolist()  # calc the stored outer hand contour points

    return {  # return dict
        "hand_mask_found": True,  # whether usable binary hand silhoutte was created
        "mask": full_mask,  # store binary hand silhouette
        "hand_contour_px": hand_contour_px,  # store outer hand contour
        "hand_mask_method": hand_mask_method,  # store segmentation method that created hand mask
        "reason": f"Hand mask created with {hand_mask_method}"  # hand mask method
    }



def _create_finger_gap_background_mask(local_points, image_shape, palm_span_px):
    """
    Mark safe visible spaces between fingers as definite background
    This is for preventing neutral shadows from "becoming" part of fingers.
    MCP=knuckle // PIP=middle joint of finger  // DIP= joint closest to fingertip // TIP=tip of finger
    """

    image_height, image_width = image_shape  # unpack returned values into image height and width in pixels so each part can be used
    gap_mask = np.zeros((image_height, image_width), dtype="uint8")  # calc crop sized mask of definite finger gap background pixels

    # Adjacent non thumb fingers. Each order of values is MCP, PIP, DIP, TIP.
    finger_pairs = [  # calc neighboring finger landmark groups used to estimate each visible gap
        ((5, 6, 7, 8), (9, 10, 11, 12)),
        ((9, 10, 11, 12), (13, 14, 15, 16)),
        ((13, 14, 15, 16), (17, 18, 19, 20)),
    ]
    # process each neighbouring pair independently because gap size and finger direction may differ across hand
    for first_chain, second_chain in finger_pairs:
        # gather each fingers four points into small arrays ordered from knucle to fingertip
        first = local_points[list(first_chain)] 
        second = local_points[list(second_chain)]  
        
        mcp_midpoint = (first[0] + second[0]) * 0.5  # calc average position of four mcp knuckles, representing upper palm center
        pip_midpoint = (first[1] + second[1]) * 0.5  # calc midpoint between neighboring PIP joints
        dip_midpoint = (first[2] + second[2]) * 0.5  # calc midpoint between beighboring DIP points 
        tip_midpoint = (first[3] + second[3]) * 0.5  # calc midpoint between neighboring fingertips

        pip_gap = float(np.linalg.norm(first[1] - second[1])) # calc distance between neighboring PIP joints
        dip_gap = float(np.linalg.norm(first[2] - second[2])) #  calc distance between neighboring DIP joints
        safe_gap = min(pip_gap, dip_gap)                      # calc smaller finger gap distance used to choose safe background strip thickness

        # When fingers touch or nearly touch, there is no visible finger gaps, hence why in that case no sure background strip is added
        if safe_gap < max(5.0, palm_span_px * 0.045):
            continue

        # Start mostly toward the PIP level, this keeps the strip away from the true finger webbing while still opening small valley the user sees
        start = (mcp_midpoint * 0.22) + (pip_midpoint * 0.78)
        # polyline list of points that describe connected path through finger
        polyline = np.array(  # create center path running along the finger
            [start, pip_midpoint, dip_midpoint, tip_midpoint],
            dtype="float32"
        )
        polyline = np.round(polyline).astype("int32").reshape(-1, 1, 2)  # prepare points for openCV: round decimal pixel coords, convert value to 32bit integers 

        thickness = int(round(np.clip(  # calc thickness in pixels of drawn mask region
            safe_gap * 0.24,
            2.0,
            max(4.0, palm_span_px * 0.065)
        )))
        # draw outer hand contour as connected line segments 
        cv2.polylines(
            gap_mask,
            [polyline],
            isClosed=False,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )
        # draw a circular measurement endpoint
        cv2.circle(
            gap_mask,
            tuple(np.round(start).astype("int32")),
            max(1, thickness // 2),
            255,
            -1,
            lineType=cv2.LINE_AA
        )

    # thumb index gap
    thumb_ip = local_points[3]   # calc thumb ip joint landmark
    thumb_tip = local_points[4]  # calc thumb tip landmark
    index_pip = local_points[6]  # calc index finger PIP joint landmark
    index_dip = local_points[7]  # calc index finger DIP joint landmark

    # calc midpoint between thumb IP joint and index finger PIP joint. This helps keep the thumb and index finger separated in the final mask
    upper_start = (thumb_ip + index_pip) * 0.5  # calc midpoint between thumb IP point and index finger PIP joint
    upper_end = (thumb_tip + index_dip) * 0.5   # calc midpoint between thumb tip and index finger DIP joint.
    thumb_gap = float(np.linalg.norm(thumb_ip - index_pip))  # calc distance between thumb and index landmarks near their gap

    # check is the gap between thumb and index finger wide enough to safely mark as background
    if thumb_gap >= max(7.0, palm_span_px * 0.065):  # draw only background strip through the thumb-index gap when gap is clearly open= atleast 7 pixels wide and 6.5% of palm's pixel span
        thumb_thickness = int(round(np.clip(  
            thumb_gap * 0.16,  # start with strip thickness 16%
            2.0,  # clamp thickness to atleast 2 pixels
            max(4.0, palm_span_px * 0.055)  # limit it with max 4 pixels, 5.5% at most of palm span
        )))
        # draw a line between two pixel points
        cv2.line(
            gap_mask,
            tuple(np.round(upper_start).astype("int32")),
            tuple(np.round(upper_end).astype("int32")),
            255,
            thumb_thickness,
            lineType=cv2.LINE_AA
        )

    return gap_mask


def _refine_grabcut_mask_against_shadows(crop,crop_mask,sure_foreground,probable_foreground,finger_gap_background,palm_span_px,):
    """remove grabcut pixels that look more like neutral paper shadow than skin"""

    # fewer than 100 fg pixels dont provide useful hand component or enough color evidence
    if np.count_nonzero(crop_mask) < 100:
        return None
    # Lab separates brightness from colour, hsv provides saturation
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype("float32") # calc image converted to lab
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype("float32") # calc image converted to HSV, color saturation

    skin_values = lab[sure_foreground > 0]  # learn skin color only from sure landmark seeds, where change of sampling paper is lowest

    # Paper samples are taken away from the landmark envelope. In a warped A4
    # image this is normally clean paper, including both lit and shadowed paper.
    envelope_size = max(9, int(round(palm_span_px * 0.16)))
    if envelope_size % 2 == 0:
        envelope_size += 1

    expanded_foreground = cv2.dilate(
        probable_foreground,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (envelope_size, envelope_size)
        ),
        iterations=1
    )
    
    paper_selection = expanded_foreground == 0

    # Explicit finger gaps are excellent local paper/shadow samples.
    paper_selection = paper_selection | (finger_gap_background > 0)
    paper_values = lab[paper_selection]
    # reliable color model need enough samples from both classes, if not skip refinement instead of making decision from few noisy pxls
    if skin_values.shape[0] < 50 or paper_values.shape[0] < 200:
        return None
    # use only a/b chroma for main comparison so darker paper shadow is not mistaken for skin, even if brightness resembles hand
    skin_ab = skin_values[:, 1:3]
    paper_ab = paper_values[:, 1:3]
    # median based centeres and bounded scales describe typical skin and paper colors while limiting influence of outliers
    skin_center, skin_scale = _robust_two_channel_model(
        skin_ab,
        scale_floor=(5.0, 5.0),
        scale_ceiling=(20.0, 20.0),
    )
    paper_center, paper_scale = _robust_two_channel_model(
        paper_ab,
        scale_floor=(3.0, 3.0),
        scale_ceiling=(14.0, 14.0),
    )

    ab = lab[:, :, 1:3]
    # calculated normalized color distance from every pixel to both learned models
    skin_distance = np.sqrt(np.sum(
        ((ab - skin_center) / skin_scale) ** 2,
        axis=2
    ))
    paper_distance = np.sqrt(np.sum(
        ((ab - paper_center) / paper_scale) ** 2,
        axis=2
    ))
    # build low saturation floor from actual skin seeds
    saturation = hsv[:, :, 1]
    skin_seed_saturation = hsv[:, :, 1][sure_foreground > 0]
    saturation_floor = max(4.0, float(np.percentile(skin_seed_saturation, 8)) * 0.30)

    # keep pixels that are closer to skin than paper, and low contrast pixels that still have enough saturation and remain close to skin model
    skin_like = (
        ((skin_distance <= (paper_distance * 1.12)) & (skin_distance < 4.2)) |
        ((saturation >= saturation_floor) & (skin_distance < 2.9))
    )
    # dilate sure foreground into protected core
    core_size = max(5, int(round(palm_span_px * 0.055)))
    if core_size % 2 == 0:
        core_size += 1

    protected_core = cv2.dilate(
        sure_foreground,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (core_size, core_size)
        ),
        iterations=1
    ) > 0
    # pixel survives only if it beloned to GC foreground and is either skin like or inside the protected core
    refined = (
        (crop_mask > 0) &
        (skin_like | protected_core)
    )
    # convert boolean result to openCV's mask and clean isolated pixels with small opening and closing operations
    refined[finger_gap_background > 0] = False
    refined[sure_foreground > 0] = True

    refined_mask = (refined.astype("uint8") * 255)
    refined_mask = cv2.morphologyEx(
        refined_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    refined_mask = cv2.morphologyEx(
        refined_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    # enforce trusted gap and hand regions one final time before returning
    refined_mask[finger_gap_background > 0] = 0
    refined_mask[sure_foreground > 0] = 255

    return refined_mask


def _robust_two_channel_model(values, scale_floor, scale_ceiling):
    """
    Returns a robust center and scale for two channel color data.
    """
    # convert incoming samples to predictable numeric type, using median to make centre resistant to few wrong skin or paper samples
    values = np.asarray(values, dtype="float32")
    center = np.median(values, axis=0)
    deviation = np.median(np.abs(values - center), axis=0) * 1.4826
    # clamp channel scales to avoid almost constant sample causing division by zero
    scale = np.clip(
        deviation,
        np.asarray(scale_floor, dtype="float32"),
        np.asarray(scale_ceiling, dtype="float32")
    )

    return center, scale


def _is_refined_component_safe(raw_component, refined_component, local_points):
    """accept shadow cleanup only if it still keeps realistic hand component"""

    # require both masks for comparison
    if raw_component is None or refined_component is None:
        return False
    # count white pixels for measuring how much of original component survived
    raw_area = int(np.count_nonzero(raw_component))
    refined_area = int(np.count_nonzero(refined_component))
    # empty masks are invalid
    if raw_area <= 0 or refined_area <= 0:
        return False
    
    # refinement should not erase most of the hand or create component larger than original
    retained_ratio = refined_area / float(raw_area)
    if retained_ratio < 0.48 or retained_ratio > 1.03:
        return False
    # check 6 central landmarks separately from all 21 landmark points: wrist, thumb MCP and four finger mcp points as strong palm anchors
    core_indices = [0, 2, 5, 9, 13, 17]
    core_support = 0
    all_support = 0
    # test 5x5 neighbourhood around every landmark because prediction can sit 1-2 pixels outside a mask edge after segmentation
    for landmark_index, point in enumerate(local_points):
        x = int(round(point[0]))
        y = int(round(point[1]))
      
        x = max(0, min(refined_component.shape[1] - 1, x))
        y = max(0, min(refined_component.shape[0] - 1, y))

        x0 = max(0, x - 2)
        x1 = min(refined_component.shape[1], x + 3)
        y0 = max(0, y - 2)
        y1 = min(refined_component.shape[0], y + 3)
        # landmark is supported when any nearby pixel remains foreground
        supported = np.any(refined_component[y0:y1, x0:x1] > 0)

        if supported:
            all_support += 1

            if landmark_index in core_indices:
                core_support += 1
    # require nearly all central landmarks and most total landmarks, allowing very minor losses but rejects cleanup that no longer resembles hand
    return core_support >= 5 and all_support >= 16

def _create_adaptive_color_fallback_mask(crop,probable_foreground,sure_foreground,):
    """
    Creates a backup hand mask from image specific skin and paper colors,
    backup used only when GC cannot create usable hand mask
    """
    # convert once into LAB and HSV: lab for skin/paper distance, saturation helps select cleaner paper samples
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype("float32")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype("float32")
    # separate brightness and saturation, clean A4 paper usually relatively bright and not strongly saturated
    lightness = lab[:, :, 0]
    saturation = hsv[:, :, 1]
    # learn hand model from trusted landmark seeds, pixels outside wider probable hand region are candidates for the paper model
    foreground_values = lab[sure_foreground > 0]
    outside_foreground = probable_foreground == 0
    # abort fallback when either class has too few examples
    if foreground_values.shape[0] < 30 or np.count_nonzero(outside_foreground) < 200:
        return None
    # measure outside region so paper samples can favor brighter and less saturated pixels while adapting to current lightning
    outside_lightness = lightness[outside_foreground]
    outside_saturation = saturation[outside_foreground]
    # prefer outside pixels above >35 brigtness and less saturated pixels
    paper_selection = (
        outside_foreground &
        (lightness >= np.percentile(outside_lightness, 35)) &
        (saturation <= np.percentile(outside_saturation, 75))
    )
    # if stricter paper rule leaves too little data, use all outside pixels rather than building model from tiny sample
    if np.count_nonzero(paper_selection) < 200:
        paper_selection = outside_foreground

    paper_values = lab[paper_selection]
    # build separate robust lab models for hand and paper, bounded scales make distance comparisons stabler
    foreground_center, foreground_scale = _robust_lab_model(
        foreground_values,
        scale_floor=(16.0, 7.0, 7.0),
        scale_ceiling=(38.0, 18.0, 18.0),
    )
    paper_center, paper_scale = _robust_lab_model(
        paper_values,
        scale_floor=(12.0, 4.0, 4.0),
        scale_ceiling=(30.0, 12.0, 12.0),
    )
    # calculate normalized distance from each pixel to learned hand and paper models, lower distance = better color match
    foreground_distance = np.sqrt(np.sum(
        ((lab - foreground_center) / foreground_scale) ** 2,
        axis=2
    ))
    paper_distance = np.sqrt(np.sum(
        ((lab - paper_center) / paper_scale) ** 2,
        axis=2
    ))
    # compare only a/b color against paper as an extra check, paper shadow could be dark but often remains close to neutral paper "chroma"
    chroma_from_paper = np.sqrt(
        (lab[:, :, 1] - paper_center[1]) ** 2 +
        (lab[:, :, 2] - paper_center[2]) ** 2
    )
    # limit candidates to dilated landmark envelope so distant objects on A4 sheet cannot become part of hand even if color similar
    envelope_size = max(11, int(round(max(crop.shape[:2]) * 0.055)))
    if envelope_size % 2 == 0:
        envelope_size += 1
    
    spatial_envelope = cv2.dilate(
        probable_foreground,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (envelope_size, envelope_size)
        ),
        iterations=1
    ) > 0

    candidate = (
        spatial_envelope &
        (foreground_distance < (paper_distance * 1.08)) &
        (foreground_distance < 3.6) &
        (
            (chroma_from_paper > 2.0) |
            (probable_foreground > 0)
        )
    )
    # trusted landmark seeds always remain hand even if color model is poor at one location, bad lightining etc..
    candidate[sure_foreground > 0] = True
    # convert to 0/255 mask and remove isolated nouse and close small holes, sure seeds are restored
    candidate_mask = (candidate.astype("uint8") * 255)
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    candidate_mask[sure_foreground > 0] = 255

    return candidate_mask


def _robust_lab_model(values, scale_floor, scale_ceiling):
    """Robust lab model, stable three channel lab color model for fallback segmenter"""
    # use float arrays and median statistics so few wrong training pixels dont pull color model far from typical LAB values
    values = np.asarray(values, dtype="float32")
    center = np.median(values, axis=0)
    deviation = np.median(np.abs(values - center), axis=0) * 1.4826  # estiamte robust per channel spread with median abs deviation
    # clamp each channel's spread to avoid zero divisions and too permissive color distances
    scale = np.clip(
        deviation,
        np.asarray(scale_floor, dtype="float32"),
        np.asarray(scale_ceiling, dtype="float32")
    )

    return center, scale

def _keep_component_supported_by_landmarks(binary_mask, local_points):
    """from several white masks, keep one supported by palm landmarks"""
    # label every connected white region, 0 = background
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )
    # count of one means only background exists, so no hand candidate
    if component_count <= 1:
        return None

    core_indices = [0, 2, 5, 9, 13, 17]
    label_support = {}
    # check which connected label appears around each core landmark
    for landmark_index in core_indices:
        x = int(round(local_points[landmark_index][0]))
        y = int(round(local_points[landmark_index][1]))

        x = max(0, min(binary_mask.shape[1] - 1, x))
        y = max(0, min(binary_mask.shape[0] - 1, y))

        # Search tiny neighborhood because a landmark may sit exactly on 1 pixel boundary
        x0 = max(0, x - 3)
        x1 = min(binary_mask.shape[1], x + 4)
        y0 = max(0, y - 3)
        y1 = min(binary_mask.shape[0], y + 4)
    
        nearby_labels = labels[y0:y1, x0:x1]
        nearby_labels = nearby_labels[nearby_labels > 0]

        if nearby_labels.size == 0:
            continue
        # give this landmark's support to label occupying most nearby pixels
        values, counts = np.unique(nearby_labels, return_counts=True)
        best_nearby_label = int(values[np.argmax(counts)])
        label_support[best_nearby_label] = label_support.get(best_nearby_label, 0) + 1
    # keep only components supported by atleast 1 core landmark, store both support count and area for ranking
    candidate_labels = []
    
    for label_value in range(1, component_count):
        area = int(stats[label_value, cv2.CC_STAT_AREA])
        support = label_support.get(label_value, 0)

        if support > 0:
            candidate_labels.append((support, area, label_value))

    if not candidate_labels:
        return None
    # sort and select component with most landmark support first
    candidate_labels.sort(reverse=True)
    _, selected_area, selected_label = candidate_labels[0]
    # reject tiny components, keep treshold of atleast 400 pixels and scales to 1,5% of crop if larger image
    minimum_area = max(400, int(binary_mask.size * 0.015))

    if selected_area < minimum_area:
        return None
    # return only chosen label as clean binary mask
    return np.where(labels == selected_label, 255, 0).astype("uint8")


def _measure_outer_palm_width_from_side_rays(hand_mask,index_mcp,middle_mcp,ring_mcp,pinky_mcp,palm_axis,palm_axis_length,landmark_span_px,scale_info,hand_length_mm,landmark_palm_width_mm,):
    """trace index and pinky sides separately to avoid connected thumb edge"""
    # average all four mcp knuckles to obtain stable centre for each sampling level
    mcp_midpoint = np.mean(
        np.array([index_mcp, middle_mcp, ring_mcp, pinky_mcp], dtype="float32"),
        axis=0
    )
    # store accepted widths + count why candidates fail
    samples = []
    rejected_anchor_count = 0
    rejected_boundary_count = 0
    rejected_ratio_count = 0

    # sample mcp row and several small shifts towards wrist, nearby levels reduce sensitivity to one knucle notch
    for wrist_shift_fraction in (0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175):
        shift_vector = -(palm_axis * palm_axis_length * wrist_shift_fraction)

        level_center = mcp_midpoint + shift_vector
        index_anchor = index_mcp + shift_vector
        pinky_anchor = pinky_mcp + shift_vector
        # build one outward vector from level centre to each side anchor
        index_direction = index_anchor - level_center
        pinky_direction = pinky_anchor - level_center

        index_length = float(np.linalg.norm(index_direction))
        pinky_length = float(np.linalg.norm(pinky_direction))
        # very short vector= usually collapsed or badly placed landmarks at this level
        if index_length < 3.0 or pinky_length < 3.0:
            rejected_anchor_count += 1
            continue
        # normalize each vector so one loop step equals about one pixel
        index_direction = index_direction / index_length
        pinky_direction = pinky_direction / pinky_length

        # If mcp anchor sits just outside tapering palm side, move it towards centre until foreground
        index_start = _move_point_inside_mask(
            hand_mask,
            index_anchor,
            level_center - index_anchor,
            maximum_distance=max(5, int(round(landmark_span_px * 0.22)))
        )
        pinky_start = _move_point_inside_mask(
            hand_mask,
            pinky_anchor,
            level_center - pinky_anchor,
            maximum_distance=max(5, int(round(landmark_span_px * 0.22)))
        )
        # both sides need valid inside start, otherwise this level cannot produce complete palm width
        if index_start is None or pinky_start is None:
            rejected_anchor_count += 1
            continue

        # limit how far each ray can travel outside its mcp anchor
        maximum_extension = max(8, int(round(landmark_span_px * 0.34)))
        #require short consecutive run of background after leaving hand
        background_run = max(2, int(round(landmark_span_px * 0.018)))

        index_edge = _trace_mask_edge_outward(
            hand_mask,
            index_start,
            index_direction,
            maximum_extension,
            background_run
        )
        pinky_edge = _trace_mask_edge_outward(
            hand_mask,
            pinky_start,
            pinky_direction,
            maximum_extension,
            background_run
        )
        # reject this level if local or outer boundary cannot be confirmed
        if index_edge is None or pinky_edge is None:
            rejected_boundary_count += 1
            continue
        # measure between 2 traced skin edges using calibrated A4 scale
        width_mm = _distance_mm_xy(
            pinky_edge,
            index_edge,
            scale_info["px_per_mm_x"],
            scale_info["px_per_mm_y"]
        )
        # remove obvious shadow hits by comparing both internal mcp width and total hand length
        width_ratio = width_mm / max(landmark_palm_width_mm, 0.001)
        hand_ratio = width_mm / max(hand_length_mm, 0.001)
        # real outer width shouldnt be smaller than inner mcp span
        if width_ratio < 0.98 or width_ratio > 1.52:
            rejected_ratio_count += 1
            continue
        # reject widths that are too narrow or broad relative to measured wrist to middletip length
        if hand_ratio < 0.22 or hand_ratio > 0.68:
            rejected_ratio_count += 1
            continue
        # preserve accepted endpoints and level position for consistency checks
        samples.append({
            "point_a": pinky_edge.astype("float32"),
            "point_b": index_edge.astype("float32"),
            "width_mm": float(width_mm),
            "wrist_shift_fraction": float(wrist_shift_fraction),
        })

    # two agreeing sideray levels are minimum, one measurement cannot prove that local edge was stable rather than accidental notch
    if len(samples) < 2:
        return {
            "measurement_ok": False,
            "palm_width_samples_mm": [
                round(float(item["width_mm"]), 1)
                for item in samples
            ],
            "reason": (
                f"Only {len(samples)} stable MCP side-ray widths were found "
                f"(anchor: {rejected_anchor_count}, "
                f"boundary: {rejected_boundary_count}, "
                f"ratio: {rejected_ratio_count})"
            )
        }
    # gather accepted widths into an array for median and outlier filtering 
    widths = np.array(
        [sample["width_mm"] for sample in samples],
        dtype="float32"
    )
    # use median as centre, keep samples within 2mm or 10%, whichever tolerance is wider
    median_width = float(np.median(widths))
    absolute_deviations = np.abs(widths - median_width)
    tolerance = max(2.0, median_width * 0.10)
    inlier_indices = np.flatnonzero(absolute_deviations <= tolerance)
    # atleast 2 samples must agree after outlier removal
    if len(inlier_indices) < 2:
        return {
            "measurement_ok": False,
            "palm_width_samples_mm": [round(float(value), 1) for value in widths.tolist()],
            "reason": "MCP side-ray widths were not consistent enough"
        }
    # final result is median of only consisten inlying widths
    inlier_widths = widths[inlier_indices]
    final_width = float(np.median(inlier_widths))
    # pick real inlier sample closest to final median so debug line corresponds to actual measured level
    selected_index = min(
        inlier_indices.tolist(),
        key=lambda index: abs(samples[index]["width_mm"] - final_width)
    )
    selected = samples[selected_index]

    return {
        "measurement_ok": True,
        "outer_palm_width_mm": round(final_width, 1),
        "palm_width_points_px": [
            [int(round(selected["point_a"][0])), int(round(selected["point_a"][1]))],
            [int(round(selected["point_b"][0])), int(round(selected["point_b"][1]))],
        ],
        "palm_width_samples_mm": [round(float(value), 1) for value in inlier_widths.tolist()],
        "palm_width_method": "outer_contour_mcp_side_rays",
        "width_definition": (
            "outer palm edges traced locally outward from the index and pinky "
            "MCP sides, excluding the distant thumb edge"
        ),
        "reason": "Outer contour width measured successfully with MCP side rays",
    }


def _move_point_inside_mask(hand_mask,point,toward_direction,maximum_distance,):
    """Move point toward palm until it reaches hand mask"""
    # movement vector points from anchor toward palm centre, length must be zero before normalization
    direction_length = float(np.linalg.norm(toward_direction))
    # near zero vector has no reliable direction and division would be unstable
    if direction_length < 1e-6:
        return None
    
    direction = toward_direction / direction_length
    image_height, image_width = hand_mask.shape[:2]
    # test original point first then move inward one pixel at a time until white handmask pixel is found
    for distance in range(maximum_distance + 1):
        candidate = point + (direction * float(distance))
        x = int(round(candidate[0]))
        y = int(round(candidate[1]))
        # ignore candidates outside image rather than indexing invalid pixels
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            continue
        # return first foreground position as safe start for outward tracing
        if hand_mask[y, x] > 0:
            return candidate.astype("float32")
    # no fg was reached withing allowed inward correction
    return None


def _trace_mask_edge_outward(hand_mask,start_point,outward_direction,maximum_distance,required_background_run,):
    """ "walk" from inside hand towards one nearby outer edge """
    # normalize outward direction
    direction_length = float(np.linalg.norm(outward_direction))
    # avoid dividing by direction that is zero
    if direction_length < 1e-6:
        return None
    
    direction = outward_direction / direction_length
    image_height, image_width = hand_mask.shape[:2]
    # track latest hand pixel and how many consecutive bg pixels have followed it, edge is accepted only after stable background run
    last_foreground = None
    background_count = 0
    # continue slitghly beyoung normal extension so required bg run can be confirmed after last fg pixel
    for distance in range(maximum_distance + required_background_run + 1):
        candidate = start_point + (direction * float(distance))
        x = int(round(candidate[0]))
        y = int(round(candidate[1]))
        # leaving image counts as bg, optherwise inspect mask pixel
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            background_count += 1
        elif hand_mask[y, x] > 0:
            last_foreground = candidate.astype("float32")
            background_count = 0
        else:
            background_count += 1
        # once enough consecutive bg is seen, return last confirmed fg point as local contour edge
        if background_count >= required_background_run:
            return last_foreground
        # if ray is still inside fg beyond its allowed local range, it could have merged shadow instead of edge
        if distance > maximum_distance and background_count == 0:
            return None

    return None


def _measure_mask_width_on_line(hand_mask, center, width_axis, landmark_span_px):
    """older backup, measure one continous handmask run across a full width line"""
    # searcb far enough in both sideways directions to cross full hand while still using later mcp based limits to reject unrealistic segments
    image_height, image_width = hand_mask.shape[:2] 
    maximum_distance = int(round(max(image_height, image_width) * 0.75))
    # create one pixel signed offset from expected centre, negative values travel one way across the palm and positive values the other way
    sample_positions = np.arange(
        -maximum_distance,
        maximum_distance + 1,
        1,
        dtype="float32"
    )
    # convert all offsets into x/y points on the width line in one vectorized operation, then round them to image coords
    sample_points = center[None, :] + (sample_positions[:, None] * width_axis[None, :])
    sample_x = np.round(sample_points[:, 0]).astype("int32")
    sample_y = np.round(sample_points[:, 1]).astype("int32")
    # mark which sampled coordinates are indise mask before array indexing
    valid = (
        (sample_x >= 0) &
        (sample_x < image_width) &
        (sample_y >= 0) &
        (sample_y < image_height)
    )
    # build one dimensional profile: true meaning that line position crosses the white hand mask
    foreground = np.zeros(len(sample_positions), dtype=bool)
    foreground[valid] = hand_mask[sample_y[valid], sample_x[valid]] > 0
    # the offset closest to zero is the expected palm centre
    center_index = int(np.argmin(np.abs(sample_positions)))
    # if the exact centre falls in a small mask hole, search nearby for closest foregorund position and limit to max 18% of mcp span
    if not foreground[center_index]:
        nearby_limit = max(5, int(round(landmark_span_px * 0.18)))
        nearby_start = max(0, center_index - nearby_limit)
        nearby_end = min(len(foreground), center_index + nearby_limit + 1)
        nearby_indices = np.flatnonzero(foreground[nearby_start:nearby_end])
        # no nearby hand pixel equals here that this line doesnt cross the expected palm
        if nearby_indices.size == 0:
            return None

        center_index = nearby_start + int(
            nearby_indices[np.argmin(
                np.abs((nearby_start + nearby_indices) - center_index)
            )]
        )
    # expand left and right through same continous foreground run, this ignores separate shadows
    left_index = center_index
    right_index = center_index

    while left_index > 0 and foreground[left_index - 1]:
        left_index -= 1

    while right_index < len(foreground) - 1 and foreground[right_index + 1]:
        right_index += 1
    # fewer thjan five sampled pixels is too short to represent a palm segment
    if right_index - left_index < 5:
        return None
    # convert run endpoints into pixel width for finals 
    point_a = sample_points[left_index]
    point_b = sample_points[right_index]
    width_px = float(np.linalg.norm(point_b - point_a))
    # valid outer contour should be atleast close to internal mcp span
    if width_px < landmark_span_px * 0.90:
        return None
    # cap run so backup line cant include shadow or a thumb if its too distant
    if width_px > landmark_span_px * 1.65:
        return None

    return {
        "point_a": point_a.astype("float32"),
        "point_b": point_b.astype("float32"),
        "width_px": width_px,
    }


def _landmarks_to_array(landmarks):
    """Convert landmark dictionaries into numpy array for vector calculations"""
    # keep only x/y pixel coords in mp index order and use float32 so subtraction, averaging and normalization work without integer shortening
    return np.array(
        [[point["x"], point["y"]] for point in landmarks],
        dtype="float32"
    )


def _distance_mm(point_a, point_b, px_per_mm_x, px_per_mm_y):
    """Convert distance between two landmark dictionaries from pixels to mm"""
    # convert horizontal (X) and vertical (y) pixel differences separately because warp may produce slightly diff x and y pixels per millimetre values
    dx_mm = (point_b["x"] - point_a["x"]) / px_per_mm_x
    dy_mm = (point_b["y"] - point_a["y"]) / px_per_mm_y
    # combine two physical components by finding straight line distance between two points, distance formula
    return math.sqrt(dx_mm ** 2 + dy_mm ** 2)


def _distance_mm_xy(point_a, point_b, px_per_mm_x, px_per_mm_y):
    """Same millimeter conversion as _distance_mm, but accepts x/y arrays"""
    # this version for accepting nympy style instead of landmark dictionaries
    dx_mm = (float(point_b[0]) - float(point_a[0])) / px_per_mm_x
    dy_mm = (float(point_b[1]) - float(point_a[1])) / px_per_mm_y
    # return physical straight line distance in mm
    return math.sqrt(dx_mm ** 2 + dy_mm ** 2)


def draw_hand_landmarks_debug(warped_a4_image, hand_result, measurement_result=None):
    """ Draws MediaPipe hand landmarks on the warped A4 image.
    Draws:hand skeleton, landmark points, selected key landmark labels, outer hand contour and palm width line when segmentation succeeds and info bar for debugging
    """
    # draw on a copy so caller's warped image remains unchanged
    output = warped_a4_image.copy()
    # successfull and failed detections use different overlay text, drawing landmarks is attempted only when primary hand exists
    if hand_result.get("hand_found", False):
        landmarks = hand_result["primary_hand"]["landmarks_px"]

        # Draw detected outer hand contour before the skeleton so skeleton lines remain visible on top of it
        if measurement_result:
            hand_contour_px = measurement_result.get("hand_contour_px", [])
            # reshape into opencv polyline format before drawing because closed contour needs atleast 3 points
            if len(hand_contour_px) >= 3:
                contour = np.array(hand_contour_px, dtype="int32").reshape(-1, 1, 2)
                cv2.polylines(
                    output,
                    [contour],
                    True,
                    (255, 180, 0),
                    2
                )

        # Draw skeleton connections first to easily see if theres wrong landmark placements
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

        # Draw every detected landmark as filled green point
        for point in landmarks:
            cv2.circle(
                output,
                (point["x"], point["y"]),
                5,
                (0, 255, 0),
                -1
            )

        # Label only the most important points to avoid covering hand with all 21 names
        key_labels = {
            0: "WRIST",
            5: "INDEX MCP",
            12: "MIDDLE TIP",
            17: "PINKY MCP",
        }
        # offset text sligthly so its not on the landmark dot
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
        # when measurement exists compare old internal mcp width with selected outer contour width
        if measurement_result:
            # Draw the old internal MCP to MCP width in yellow for comparison
            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]

            cv2.line(
                output,
                (index_mcp["x"], index_mcp["y"]),
                (pinky_mcp["x"], pinky_mcp["y"]),
                (0, 220, 255),
                2
            )

            # Draw new outer contour palm width when segmentation passed
            palm_width_points = measurement_result.get("palm_width_points_px")
            # require exactly two endpoints before drawing selected width
            if palm_width_points and len(palm_width_points) == 2:
                point_a = tuple(map(int, palm_width_points[0]))
                point_b = tuple(map(int, palm_width_points[1]))

                cv2.line(
                    output,
                    point_a,
                    point_b,
                    (255, 0, 255),
                    3
                )
                cv2.circle(output, point_a, 6, (255, 0, 255), -1)
                cv2.circle(output, point_b, 6, (255, 0, 255), -1)
        # build readable status lines for information bar
        primary_hand = hand_result["primary_hand"]

        info_lines = [
            "MEDIAPIPE HAND FOUND",
            f"Handedness: {primary_hand.get('handedness', 'unknown')} "
            f"({primary_hand.get('handedness_score', 0):.2f})",
        ]
        # add numeric measurements, chosen method and mask status when avaialble
        if measurement_result:
            info_lines.append(
                f"Approx length: {measurement_result['approx_hand_length_mm']:.1f} mm"
            )
            info_lines.append(
                f"Palm width: {measurement_result['approx_palm_width_mm']:.1f} mm "
                f"({measurement_result.get('palm_width_method', 'unknown')})"
            )
            info_lines.append(
                f"Landmark width comparison: "
                f"{measurement_result['landmark_palm_width_mm']:.1f} mm"
            )
            info_lines.append(
                f"Hand mask: {measurement_result.get('hand_mask_found', False)} "
                f"({measurement_result.get('hand_mask_method', 'unknown')})"
            )

            if measurement_result.get("segmentation_reason"):
                info_lines.append(
                    f"Mask/width status: {measurement_result['segmentation_reason']}"
                )

    else: # failed detection still returns debug image with its reason
        info_lines = [
            "HAND NOT FOUND",
            f"Reason: {hand_result.get('reason', 'unknown')}"
        ]
     # add text panel after all iamge drawings so it stays readable
    _draw_info_bar(output, info_lines)
    # return image with text panel and infos for debug
    return output


def _draw_info_bar(image, lines):
    """Draws transparent information bar at the top of the image"""
    # reserve one 28pixel row per text line + extra vertical padding
    bar_h = 28 * (len(lines) + 1)
    # draw dark rectangle on copy, then blend it with image to create transparency without darkening rest of its frame
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1], bar_h),
        (20, 20, 20),
        -1
    )
    # 65% dark overlay and 35% original image inside the rectangle
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    # start first text baseline below the top edge
    y = 22
    # draw each status line and move down by fixed amount for next row
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
        y += 26 # keep lines separated while fitting insidethe 28 pixel row height