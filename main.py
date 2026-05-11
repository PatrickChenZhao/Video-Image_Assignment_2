"""
Phase 3: dual-hand multimodal virtual Xbox controller.

Run:
    python main.py

The runtime loop is optimized for low latency: no CSV logging and no KNN
inference are performed during live control.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import vgamepad as vg


mp_holistic = mp.solutions.holistic


# Right-hand SVM button recognition settings.
DEBOUNCE_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.80
NO_HAND_RELEASE_FRAMES = 3
RIGHT_HAND_SVM_INTERVAL_FRAMES = 2

# Left-arm shoulder-anchored virtual joystick settings.
SAVJ_DEAD_ZONE_PIXELS = 40
SAVJ_R_MAX_PIXELS = 150
XINPUT_AXIS_MAX = 32767
OPPOSITE_CAMERA_HAND_SWAP = True
CONTROL_LEFT_SHOULDER_LANDMARK = 12
CONTROL_LEFT_WRIST_LANDMARK = 16

PAD_MAPPING = {
    "Paper": {
        "button": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        "display": "Paper: A",
    },
    "Scissors": {
        "button": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        "display": "Scissors: X",
    },
    "Thumb": {
        "button": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "display": "Thumb: B",
    },
    "Rock": {
        "button": None,
        "display": "Rock: RELEASE",
    },
}

gamepad = vg.VX360Gamepad()

CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 424
CAMERA_FRAME_HEIGHT = 240
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

SAVJ_SHOULDER_COLOR = (255, 0, 0)
SAVJ_WRIST_COLOR = (0, 0, 255)
SAVJ_ARM_LINE_COLOR = (0, 255, 0)

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
    return scaler, svm_model


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
    prediction = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return prediction, confidence


def predict_right_hand_gesture(
    right_hand_landmarks,
    scaler,
    svm_model,
) -> tuple[Optional[str], Optional[float], bool]:
    """Classify the right hand with the SVM model."""
    if right_hand_landmarks is None:
        return None, None, False

    features = extract_relative_features_from_landmarks(right_hand_landmarks)
    feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)

    svm_prediction, confidence = predict_with_confidence(svm_model, scaled_features)
    return svm_prediction, confidence, True


def get_stable_gesture(prediction_history: deque[str]) -> Optional[str]:
    """Return a gesture only when the debounce queue is full and unanimous."""
    if len(prediction_history) < DEBOUNCE_FRAMES:
        return None

    first_prediction = prediction_history[0]
    if all(prediction == first_prediction for prediction in prediction_history):
        return first_prediction

    return None


def release_pressed_button(pressed_button) -> None:
    """Release the currently held gamepad button and update the device."""
    if pressed_button is None:
        return

    try:
        gamepad.release_button(button=pressed_button)
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to release gamepad button {pressed_button}: {exc}")


def apply_right_hand_state(
    gesture: str,
    current_gesture: Optional[str],
    pressed_button,
) -> tuple[Optional[str], object]:
    """Apply a debounced right-hand gesture to the mapped Xbox button state."""
    if gesture == current_gesture:
        return current_gesture, pressed_button

    release_pressed_button(pressed_button)

    mapping = PAD_MAPPING.get(gesture)
    if mapping is None:
        return None, None

    target_button = mapping["button"]
    if target_button is None:
        return gesture, None

    try:
        gamepad.press_button(button=target_button)
        gamepad.update()
        return gesture, target_button
    except Exception as exc:
        print(f"[Warning] Failed to press gamepad button {target_button}: {exc}")
        return None, None


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
    """Return left shoulder and left wrist pixel coordinates."""
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


def get_savj_anchor_points(pose_landmarks, frame_width: int, frame_height: int) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Return controller-left shoulder and wrist points in pixel coordinates."""
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


def draw_holistic_landmarks(frame, results) -> None:
    """Draw only the lightweight SAVJ visualization."""
    draw_savj_visual(frame, results.pose_landmarks)


def get_controller_right_hand_landmarks(results):
    """Return the physical right hand landmarks after opposite-camera hand swapping."""
    if OPPOSITE_CAMERA_HAND_SWAP:
        return results.left_hand_landmarks

    return results.right_hand_landmarks


def draw_status(
    frame,
    raw_gesture: Optional[str],
    confidence: Optional[float],
    stable_gesture: Optional[str],
    has_right_hand: bool,
    axis_x: int,
    axis_y: int,
    has_left_arm: bool,
) -> None:
    """Draw runtime status text on the camera frame."""
    if stable_gesture in PAD_MAPPING:
        confidence_text = "" if confidence is None else f" ({confidence * 100:.1f}%)"
        status_text = f"Right: {PAD_MAPPING[stable_gesture]['display']}{confidence_text}"
        color = (0, 255, 0)
    elif raw_gesture in PAD_MAPPING and confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        status_text = f"Right: Low Confidence {raw_gesture} ({confidence * 100:.1f}%)"
        color = (0, 165, 255)
    elif raw_gesture in PAD_MAPPING:
        confidence_text = "" if confidence is None else f" ({confidence * 100:.1f}%)"
        status_text = f"Right: Detecting {raw_gesture}{confidence_text}"
        color = (0, 255, 255)
    elif not has_right_hand:
        status_text = "Right: No Hand"
        color = (0, 0, 255)
    else:
        status_text = "Right: Unknown"
        color = (0, 0, 255)

    if has_left_arm:
        left_text = f"Left Stick: joy_x={axis_x} joy_y={axis_y}"
        left_color = (0, 255, 0)
    else:
        left_text = "Left Stick: No Pose"
        left_color = (0, 0, 255)

    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, left_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    """Run real-time Holistic inference and virtual gamepad control."""
    scaler, svm_model = load_runtime_objects()

    prediction_history: deque[str] = deque(maxlen=DEBOUNCE_FRAMES)
    current_gesture: Optional[str] = None
    pressed_button = None
    no_hand_frames = 0
    frame_index = 0
    raw_gesture: Optional[str] = None
    confidence: Optional[float] = None
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

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic_detector.process(image_rgb)
                    draw_holistic_landmarks(frame, results)

                    axis_x, axis_y, has_left_arm = update_savj(frame, results.pose_landmarks)
                    controller_right_hand_landmarks = get_controller_right_hand_landmarks(results)
                    if frame_index % RIGHT_HAND_SVM_INTERVAL_FRAMES == 0:
                        raw_gesture, confidence, has_right_hand = predict_right_hand_gesture(
                            right_hand_landmarks=controller_right_hand_landmarks,
                            scaler=scaler,
                            svm_model=svm_model,
                        )
                    elif controller_right_hand_landmarks is None:
                        raw_gesture, confidence, has_right_hand = None, None, False
                    else:
                        has_right_hand = True
                except Exception as exc:
                    print(f"[Warning] Current frame prediction failed: {exc}")
                    axis_x, axis_y, has_left_arm = 0, 0, False
                    raw_gesture, confidence, has_right_hand = None, None, False

                stable_gesture = None
                is_confident = (
                    has_right_hand
                    and raw_gesture in PAD_MAPPING
                    and confidence is not None
                    and confidence >= CONFIDENCE_THRESHOLD
                )

                if is_confident:
                    no_hand_frames = 0
                    prediction_history.append(raw_gesture)
                    stable_gesture = get_stable_gesture(prediction_history)

                    if stable_gesture is not None:
                        current_gesture, pressed_button = apply_right_hand_state(
                            gesture=stable_gesture,
                            current_gesture=current_gesture,
                            pressed_button=pressed_button,
                        )
                elif has_right_hand:
                    prediction_history.clear()
                    no_hand_frames = 0
                else:
                    prediction_history.clear()
                    no_hand_frames += 1

                    if no_hand_frames >= NO_HAND_RELEASE_FRAMES:
                        release_pressed_button(pressed_button)
                        current_gesture = None
                        pressed_button = None

                draw_status(
                    frame=frame,
                    raw_gesture=raw_gesture,
                    confidence=confidence,
                    stable_gesture=stable_gesture,
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
        release_pressed_button(pressed_button)
        reset_gamepad()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera, windows, joystick, and buttons have been released.")


if __name__ == "__main__":
    main()
