"""
Phase 3: dual-hand multimodal virtual Xbox controller.

Run:
    python main.py

The runtime loop is optimized for low latency: no CSV logging and no KNN
inference are performed during live control.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import vgamepad as vg


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Dual-hand SVM button recognition settings.
DEBOUNCE_FRAMES = 3
HAND_SVM_INTERVAL_FRAMES = 2

# Left-arm shoulder-anchored virtual joystick settings.
SAVJ_DEAD_ZONE_PIXELS = 40
SAVJ_R_MAX_PIXELS = 150
XINPUT_AXIS_MAX = 32767
CONTROL_LEFT_SHOULDER_LANDMARK = 12
CONTROL_LEFT_WRIST_LANDMARK = 16

PAD_KIND_BUTTON = "button"
PAD_KIND_DPAD = "dpad"
PAD_KIND_TRIGGER = "trigger"
PAD_KIND_RELEASE = "release"
HAND_LEFT = "left"
HAND_RIGHT = "right"
TRIGGER_LEFT = "left"
TRIGGER_RIGHT = "right"

PAD_MAPPING = {
    "clasp": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "down": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
    "home": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
    "left": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
    "paper": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "rb": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
    "right": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    "rt": None,
    "scissors": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "thumb": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "up": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
    "rock": None,
    "lb": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    "leftrunning": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
    "lt": None,
}

PAD_ACTIONS = {
    # Physical right hand: action buttons, D-pad, right shoulder, RT, release.
    "clasp": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["clasp"],
        "display": "Right Y",
    },
    "down": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_DPAD,
        "button": PAD_MAPPING["down"],
        "display": "Right D-pad Down",
    },
    "home": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["home"],
        "display": "Right Start",
    },
    "left": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_DPAD,
        "button": PAD_MAPPING["left"],
        "display": "Right D-pad Left",
    },
    "paper": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["paper"],
        "display": "Right A",
    },
    "rb": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["rb"],
        "display": "Right RB",
    },
    "right": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_DPAD,
        "button": PAD_MAPPING["right"],
        "display": "Right D-pad Right",
    },
    "rt": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_TRIGGER,
        "trigger": TRIGGER_RIGHT,
        "display": "Right RT",
    },
    "scissors": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["scissors"],
        "display": "Right X",
    },
    "thumb": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["thumb"],
        "display": "Right B",
    },
    "up": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_DPAD,
        "button": PAD_MAPPING["up"],
        "display": "Right D-pad Up",
    },
    "rock": {
        "hand": HAND_RIGHT,
        "kind": PAD_KIND_RELEASE,
        "button": None,
        "display": "Right Release",
    },
    # Physical left hand: left shoulder, left stick click, LT.
    "lb": {
        "hand": HAND_LEFT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["lb"],
        "display": "Left LB",
    },
    "leftrunning": {
        "hand": HAND_LEFT,
        "kind": PAD_KIND_BUTTON,
        "button": PAD_MAPPING["leftrunning"],
        "display": "Left Stick Click",
    },
    "lt": {
        "hand": HAND_LEFT,
        "kind": PAD_KIND_TRIGGER,
        "trigger": TRIGGER_LEFT,
        "display": "Left LT",
    },
}

LEFT_GESTURES = {
    gesture for gesture, mapping in PAD_ACTIONS.items() if mapping["hand"] == HAND_LEFT
}
RIGHT_GESTURES = {
    gesture for gesture, mapping in PAD_ACTIONS.items() if mapping["hand"] == HAND_RIGHT
}

gamepad = vg.VX360Gamepad()
left_gesture_queue: deque[Optional[str]] = deque(maxlen=DEBOUNCE_FRAMES)
right_gesture_queue: deque[Optional[str]] = deque(maxlen=DEBOUNCE_FRAMES)
current_left_state: Optional[str] = None
current_right_state: Optional[str] = None

CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 424
CAMERA_FRAME_HEIGHT = 240
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

SAVJ_SHOULDER_COLOR = (255, 0, 0)
SAVJ_WRIST_COLOR = (0, 0, 255)
SAVJ_ARM_LINE_COLOR = (0, 255, 0)
LEFT_HAND_CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(255, 128, 0), thickness=2)
RIGHT_HAND_CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
HAND_LANDMARK_STYLE = mp_drawing_styles.get_default_hand_landmarks_style()

ROOT_DIR = Path(__file__).resolve().parent
SCALER_PATH = ROOT_DIR / "scaler.pkl"
SVM_MODEL_PATH = ROOT_DIR / "svm_model.pkl"


def load_runtime_objects():
    """Load the scaler and SVM model saved by train_model.py."""
    required_files = [SCALER_PATH, SVM_MODEL_PATH]
    missing_files = [file_path for file_path in required_files if not file_path.exists()]
    if missing_files:
        missing_text = "\n".join(str(file_path) for file_path in missing_files)
        raise FileNotFoundError(
            "Missing runtime files:\n"
            f"{missing_text}\n"
            "Run Phase 2 first: python train_model.py"
        )

    scaler = joblib.load(SCALER_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    model_labels = {normalize_gesture_label(str(label)) for label in svm_model.classes_}
    expected_labels = set(PAD_MAPPING)
    if model_labels != expected_labels:
        missing_labels = sorted(expected_labels - model_labels)
        extra_labels = sorted(model_labels - expected_labels)
        raise ValueError(
            "svm_model.pkl labels do not match PAD_MAPPING. "
            f"Missing: {missing_labels}; Extra: {extra_labels}. "
            "Rerun: python train_model.py"
        )
    return scaler, svm_model


def normalize_gesture_label(label: str) -> str:
    """Normalize legacy title-case model labels into runtime gesture keys."""
    return label.strip().lower()


def extract_relative_features_from_landmarks(hand_landmarks) -> list[float]:
    """
    Extract 60 scale-normalized relative hand-landmark features.

    Landmark 0 is used as the origin, and the distance from Landmark 0 to
    Landmark 9 is used as the scale reference.
    """
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]
    middle_finger_mcp = landmarks[9]
    base_distance = (
        (middle_finger_mcp.x - wrist.x) ** 2
        + (middle_finger_mcp.y - wrist.y) ** 2
        + (middle_finger_mcp.z - wrist.z) ** 2
    ) ** 0.5
    if base_distance < 1e-6:
        base_distance = 1e-6

    features: list[float] = []
    for landmark in landmarks[1:]:
        features.extend(
            [
                (landmark.x - wrist.x) / base_distance,
                (landmark.y - wrist.y) / base_distance,
                (landmark.z - wrist.z) / base_distance,
            ]
        )

    if len(features) != 60:
        raise ValueError(f"Invalid runtime feature length. Expected 60, got {len(features)}")

    return features


def predict_with_confidence(model, scaled_features) -> tuple[str, float]:
    """Predict with the SVM model and return the highest class probability."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "svm_model.pkl does not support predict_proba(). "
            "Run Phase 2 again: python train_model.py"
        )

    probabilities = model.predict_proba(scaled_features)[0]
    best_index = int(np.argmax(probabilities))
    prediction = normalize_gesture_label(str(model.classes_[best_index]))
    confidence = float(probabilities[best_index])
    return prediction, confidence


def predict_hand_gesture(
    hand_landmarks,
    scaler,
    svm_model,
) -> tuple[Optional[str], Optional[float], bool]:
    """Classify one detected hand with the SVM model."""
    if hand_landmarks is None:
        return None, None, False

    features = extract_relative_features_from_landmarks(hand_landmarks)
    feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)

    svm_prediction, confidence = predict_with_confidence(svm_model, scaled_features)
    return svm_prediction, confidence, True


def get_stable_state(prediction_history: deque[Optional[str]]) -> tuple[bool, Optional[str]]:
    """Return a debounced state when a full queue has a majority winner."""
    if len(prediction_history) < DEBOUNCE_FRAMES:
        return False, None

    state, count = Counter(prediction_history).most_common(1)[0]
    if count > len(prediction_history) // 2:
        return True, state

    return False, None


def set_trigger(trigger: str, value: float) -> None:
    """Set a trigger axis and update the virtual gamepad immediately."""
    try:
        if trigger == TRIGGER_LEFT:
            gamepad.left_trigger_float(value)
        elif trigger == TRIGGER_RIGHT:
            gamepad.right_trigger_float(value)
        else:
            raise ValueError(f"Unknown trigger axis: {trigger}")
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to set {trigger} trigger to {value}: {exc}")


def press_mapped_button(button) -> None:
    """Press a regular or D-pad button and update the virtual gamepad."""
    try:
        gamepad.press_button(button=button)
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to press gamepad button {button}: {exc}")


def release_mapped_button(button) -> None:
    """Release a regular or D-pad button and update the virtual gamepad."""
    try:
        gamepad.release_button(button=button)
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to release gamepad button {button}: {exc}")


def release_state(gesture: Optional[str]) -> None:
    """Release the controls held by one mapped gesture."""
    if gesture is None:
        return

    mapping = PAD_ACTIONS.get(gesture)
    if mapping is None:
        return

    if mapping["kind"] == PAD_KIND_TRIGGER:
        set_trigger(mapping["trigger"], 0.0)
    elif mapping["kind"] in {PAD_KIND_BUTTON, PAD_KIND_DPAD}:
        release_mapped_button(mapping["button"])


def press_state(gesture: str) -> None:
    """Press the controls represented by one mapped gesture."""
    mapping = PAD_ACTIONS.get(gesture)
    if mapping is None:
        return

    if mapping["kind"] == PAD_KIND_TRIGGER:
        set_trigger(mapping["trigger"], 1.0)
    elif mapping["kind"] in {PAD_KIND_BUTTON, PAD_KIND_DPAD}:
        press_mapped_button(mapping["button"])


def release_all_hand_controls(hand: str) -> None:
    """Release every button and trigger assigned to one physical hand."""
    for mapping in PAD_ACTIONS.values():
        if mapping["hand"] != hand:
            continue

        if mapping["kind"] == PAD_KIND_TRIGGER:
            set_trigger(mapping["trigger"], 0.0)
        elif mapping["kind"] in {PAD_KIND_BUTTON, PAD_KIND_DPAD}:
            release_mapped_button(mapping["button"])


def apply_left_hand_state(new_state: Optional[str]) -> None:
    """Apply debounced physical-left-hand state changes."""
    global current_left_state

    if new_state == current_left_state:
        return

    release_state(current_left_state)

    if new_state is not None:
        press_state(new_state)

    current_left_state = new_state


def apply_right_hand_state(new_state: Optional[str]) -> None:
    """Apply debounced physical-right-hand state changes."""
    global current_right_state

    if new_state == current_right_state:
        return

    if new_state == "rock":
        release_all_hand_controls(HAND_RIGHT)
        current_right_state = new_state
        return

    release_state(current_right_state)

    if new_state is not None:
        press_state(new_state)

    current_right_state = new_state


def reset_gamepad() -> None:
    """Reset the virtual gamepad and update the device immediately."""
    try:
        gamepad.reset()
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to reset gamepad: {exc}")


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a float to a closed interval."""
    return max(minimum, min(maximum, value))


def map_relative_position_to_axis(relative_position: float) -> int:
    """Map a clipped shoulder-relative pixel position to the XInput axis range."""
    clipped_position = clamp(relative_position, -SAVJ_R_MAX_PIXELS, SAVJ_R_MAX_PIXELS)
    axis_value = int((clipped_position / SAVJ_R_MAX_PIXELS) * XINPUT_AXIS_MAX)
    return int(clamp(axis_value, -XINPUT_AXIS_MAX, XINPUT_AXIS_MAX))


def send_left_joystick(axis_x: int, axis_y: int) -> None:
    """Send the left-stick axis values through vgamepad."""
    try:
        gamepad.left_joystick(x_value=axis_x, y_value=axis_y)
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to update left joystick: {exc}")


def get_left_shoulder_wrist_pixels(
    pose_landmarks,
    frame_width: int,
    frame_height: int,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Return physical-left shoulder and wrist pixel coordinates."""
    if pose_landmarks is None:
        return None

    landmarks = pose_landmarks.landmark
    left_shoulder = landmarks[CONTROL_LEFT_SHOULDER_LANDMARK]
    left_wrist = landmarks[CONTROL_LEFT_WRIST_LANDMARK]
    shoulder_x = left_shoulder.x * frame_width
    shoulder_y = left_shoulder.y * frame_height
    wrist_x = left_wrist.x * frame_width
    wrist_y = left_wrist.y * frame_height
    return (shoulder_x, shoulder_y), (wrist_x, wrist_y)


def update_savj(frame, pose_landmarks) -> tuple[int, int, bool]:
    """Update the left stick from the real-time wrist position relative to the shoulder."""
    frame_height, frame_width = frame.shape[:2]
    shoulder_wrist_pixels = get_left_shoulder_wrist_pixels(pose_landmarks, frame_width, frame_height)

    if shoulder_wrist_pixels is None:
        send_left_joystick(0, 0)
        return 0, 0, False

    (shoulder_x, shoulder_y), (wrist_x, wrist_y) = shoulder_wrist_pixels
    rel_x = wrist_x - shoulder_x
    rel_y = shoulder_y - wrist_y
    distance = math.sqrt(rel_x**2 + rel_y**2)

    if distance < SAVJ_DEAD_ZONE_PIXELS:
        joy_x = 0
        joy_y = 0
    else:
        joy_x = map_relative_position_to_axis(rel_x)
        joy_y = map_relative_position_to_axis(rel_y)

    send_left_joystick(joy_x, joy_y)
    return joy_x, joy_y, True


def get_savj_anchor_points(
    pose_landmarks,
    frame_width: int,
    frame_height: int,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Return physical-left shoulder and wrist points in pixel coordinates."""
    shoulder_wrist_pixels = get_left_shoulder_wrist_pixels(pose_landmarks, frame_width, frame_height)
    if shoulder_wrist_pixels is None:
        return None

    (shoulder_x, shoulder_y), (wrist_x, wrist_y) = shoulder_wrist_pixels
    shoulder_point = (int(shoulder_x), int(shoulder_y))
    wrist_point = (int(wrist_x), int(wrist_y))
    return shoulder_point, wrist_point


def draw_savj_visual(frame, pose_landmarks) -> None:
    """Draw only the SAVJ shoulder point, wrist point, and arm vector line."""
    frame_height, frame_width = frame.shape[:2]
    anchor_points = get_savj_anchor_points(pose_landmarks, frame_width, frame_height)
    if anchor_points is None:
        return

    shoulder_point, wrist_point = anchor_points
    cv2.line(frame, shoulder_point, wrist_point, SAVJ_ARM_LINE_COLOR, 3, cv2.LINE_AA)
    cv2.circle(frame, shoulder_point, 8, SAVJ_SHOULDER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, wrist_point, 8, SAVJ_WRIST_COLOR, -1, cv2.LINE_AA)


def get_physical_left_hand_landmarks(results):
    """Return the physical left hand after the mirrored-camera hand swap."""
    return results.right_hand_landmarks


def get_physical_right_hand_landmarks(results):
    """Return the physical right hand after the mirrored-camera hand swap."""
    return results.left_hand_landmarks


def draw_holistic_landmarks(frame, results) -> None:
    """Draw SAVJ plus both controller hand landmarks."""
    draw_savj_visual(frame, results.pose_landmarks)

    physical_left_hand_landmarks = get_physical_left_hand_landmarks(results)
    if physical_left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            physical_left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            HAND_LANDMARK_STYLE,
            LEFT_HAND_CONNECTION_STYLE,
        )

    physical_right_hand_landmarks = get_physical_right_hand_landmarks(results)
    if physical_right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            physical_right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            HAND_LANDMARK_STYLE,
            RIGHT_HAND_CONNECTION_STYLE,
        )


def resolve_left_candidate(
    raw_gesture: Optional[str],
    has_hand: bool,
) -> Optional[str]:
    """Map a physical-left prediction into a debounced state candidate."""
    if has_hand and raw_gesture in LEFT_GESTURES:
        return raw_gesture

    return None


def resolve_right_candidate(
    raw_gesture: Optional[str],
    has_hand: bool,
) -> Optional[str]:
    """Map a physical-right prediction into a debounced state candidate."""
    if has_hand and raw_gesture in RIGHT_GESTURES:
        return raw_gesture

    return None


def update_debounced_state(
    queue: deque[Optional[str]],
    candidate: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Push a candidate into one hand's debounce queue and return stable output."""
    queue.append(candidate)
    return get_stable_state(queue)


def format_hand_status(
    hand_label: str,
    active_state: Optional[str],
    has_hand: bool,
) -> tuple[str, tuple[int, int, int]]:
    """Build one stable hand status line for the camera overlay."""
    if active_state is None and not has_hand:
        return f"{hand_label}: No Hand", (0, 0, 255)

    if active_state is not None:
        mapping = PAD_ACTIONS.get(active_state)
        display = mapping["display"] if mapping else active_state
        return f"{hand_label}: Holding {display}", (0, 255, 0)

    return f"{hand_label}: Release", (0, 255, 0)


def draw_status(
    frame,
    has_left_hand: bool,
    has_right_hand: bool,
    axis_x: int,
    axis_y: int,
    has_left_arm: bool,
) -> None:
    """Draw runtime status text on the camera frame."""
    left_text, left_color = format_hand_status(
        hand_label="Left",
        active_state=current_left_state,
        has_hand=has_left_hand,
    )
    right_text, right_color = format_hand_status(
        hand_label="Right",
        active_state=current_right_state,
        has_hand=has_right_hand,
    )

    if has_left_arm:
        stick_text = f"Left Stick: joy_x={axis_x} joy_y={axis_y}"
        stick_color = (0, 255, 0)
    else:
        stick_text = "Left Stick: No Pose"
        stick_color = (0, 0, 255)

    cv2.putText(frame, left_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, left_color, 2, cv2.LINE_AA)
    cv2.putText(frame, right_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, right_color, 2, cv2.LINE_AA)
    cv2.putText(frame, stick_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stick_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit", (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    """Run real-time Holistic inference and virtual gamepad control."""
    global current_left_state, current_right_state

    scaler, svm_model = load_runtime_objects()
    left_gesture_queue.clear()
    right_gesture_queue.clear()
    current_left_state = None
    current_right_state = None

    frame_index = 0
    left_raw_gesture: Optional[str] = None
    has_left_hand = False
    right_raw_gesture: Optional[str] = None
    has_right_hand = False

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {CAMERA_INDEX}")

    print("Phase 3 dual-hand gamepad controller started. Press q to quit.")
    print("Runtime logging and KNN inference are disabled for lower latency.")

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as holistic_detector:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[Warning] Failed to read a camera frame; skipping this frame.")
                    continue

                frame = cv2.flip(frame, 1)
                frame_index += 1
                left_stable_ready = False
                left_stable_state: Optional[str] = None
                right_stable_ready = False
                right_stable_state: Optional[str] = None

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic_detector.process(image_rgb)
                    draw_holistic_landmarks(frame, results)

                    axis_x, axis_y, has_left_arm = update_savj(frame, results.pose_landmarks)
                    physical_left_landmarks = get_physical_left_hand_landmarks(results)
                    physical_right_landmarks = get_physical_right_hand_landmarks(results)

                    should_predict = frame_index % HAND_SVM_INTERVAL_FRAMES == 0
                    if should_predict:
                        left_raw_gesture, _, has_left_hand = predict_hand_gesture(
                            hand_landmarks=physical_left_landmarks,
                            scaler=scaler,
                            svm_model=svm_model,
                        )
                        right_raw_gesture, _, has_right_hand = predict_hand_gesture(
                            hand_landmarks=physical_right_landmarks,
                            scaler=scaler,
                            svm_model=svm_model,
                        )

                        left_candidate = resolve_left_candidate(
                            raw_gesture=left_raw_gesture,
                            has_hand=has_left_hand,
                        )
                        right_candidate = resolve_right_candidate(
                            raw_gesture=right_raw_gesture,
                            has_hand=has_right_hand,
                        )

                        left_stable_ready, left_stable_state = update_debounced_state(
                            left_gesture_queue,
                            left_candidate,
                        )
                        right_stable_ready, right_stable_state = update_debounced_state(
                            right_gesture_queue,
                            right_candidate,
                        )

                        if left_stable_ready:
                            apply_left_hand_state(left_stable_state)
                        if right_stable_ready:
                            apply_right_hand_state(right_stable_state)
                    else:
                        has_left_hand = physical_left_landmarks is not None
                        has_right_hand = physical_right_landmarks is not None
                except Exception as exc:
                    print(f"[Warning] Current frame prediction failed: {exc}")
                    axis_x, axis_y, has_left_arm = 0, 0, False
                    left_raw_gesture, has_left_hand = None, False
                    right_raw_gesture, has_right_hand = None, False

                    left_stable_ready, left_stable_state = update_debounced_state(
                        left_gesture_queue,
                        None,
                    )
                    right_stable_ready, right_stable_state = update_debounced_state(
                        right_gesture_queue,
                        None,
                    )
                    if left_stable_ready:
                        apply_left_hand_state(left_stable_state)
                    if right_stable_ready:
                        apply_right_hand_state(right_stable_state)

                draw_status(
                    frame=frame,
                    has_left_hand=has_left_hand,
                    has_right_hand=has_right_hand,
                    axis_x=axis_x,
                    axis_y=axis_y,
                    has_left_arm=has_left_arm,
                )
                cv2.imshow("Dual-Hand Holistic Game Controller", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C; shutting down.")
    finally:
        release_state(current_left_state)
        release_state(current_right_state)
        current_left_state = None
        current_right_state = None
        reset_gamepad()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera, windows, joystick, triggers, and buttons have been released.")


if __name__ == "__main__":
    main()
