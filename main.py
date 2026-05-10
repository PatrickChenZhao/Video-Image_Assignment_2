"""
Phase 3: dual-hand multimodal virtual Xbox controller.

Run:
    python main.py

Logs:
    1. latency_log.csv records per-frame SVM/KNN inference latency.
    2. resource_log.csv records Python process CPU and memory usage once per second.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import psutil
import vgamepad as vg


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# Right-hand SVM button recognition settings.
DEBOUNCE_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.80
NO_HAND_RELEASE_FRAMES = 3

# Left-arm shoulder-anchored virtual joystick settings.
CALIBRATION_FRAMES = 90
SAVJ_DEAD_ZONE_PIXELS = 10
SAVJ_R_MAX_PIXELS = 150
XINPUT_AXIS_MAX = 32767
OPPOSITE_CAMERA_HAND_SWAP = True

if OPPOSITE_CAMERA_HAND_SWAP:
    CONTROL_LEFT_SHOULDER_LANDMARK = 12
    CONTROL_LEFT_WRIST_LANDMARK = 16
else:
    CONTROL_LEFT_SHOULDER_LANDMARK = 11
    CONTROL_LEFT_WRIST_LANDMARK = 15

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
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

HAND_LANDMARK_STYLE = mp_drawing_styles.get_default_hand_landmarks_style()
GREEN_HAND_CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
SAVJ_SHOULDER_COLOR = (255, 0, 0)
SAVJ_WRIST_COLOR = (0, 0, 255)
SAVJ_ARM_LINE_COLOR = (0, 255, 0)

ROOT_DIR = Path(__file__).resolve().parent
SCALER_PATH = ROOT_DIR / "scaler.pkl"
SVM_MODEL_PATH = ROOT_DIR / "svm_model.pkl"
KNN_MODEL_PATH = ROOT_DIR / "knn_model.pkl"
LATENCY_LOG_PATH = ROOT_DIR / "latency_log.csv"
RESOURCE_LOG_PATH = ROOT_DIR / "resource_log.csv"


@dataclass
class SavjState:
    """Calibration state for the shoulder-anchored virtual joystick."""

    sample_count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    neutral_x: Optional[float] = None
    neutral_y: Optional[float] = None

    @property
    def is_calibrated(self) -> bool:
        return self.neutral_x is not None and self.neutral_y is not None

    def add_sample(self, vector_x: float, vector_y: float) -> None:
        if self.is_calibrated:
            return

        self.sample_count += 1
        self.sum_x += vector_x
        self.sum_y += vector_y

        if self.sample_count >= CALIBRATION_FRAMES:
            self.neutral_x = self.sum_x / self.sample_count
            self.neutral_y = self.sum_y / self.sample_count
            print(f"SAVJ neutral vector calibrated: ({self.neutral_x:.2f}, {self.neutral_y:.2f})")


def current_timestamp() -> str:
    """Return an ISO timestamp suitable for CSV logging."""
    return datetime.now().isoformat(timespec="milliseconds")


def load_runtime_objects():
    """Load the scaler, SVM model, and KNN model saved by train_model.py."""
    required_files = [SCALER_PATH, SVM_MODEL_PATH, KNN_MODEL_PATH]
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
    knn_model = joblib.load(KNN_MODEL_PATH)
    return scaler, svm_model, knn_model


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


def time_model_prediction(model, scaled_features) -> tuple[str, float]:
    """Run one prediction and return the prediction and elapsed milliseconds."""
    start_time = time.perf_counter()
    prediction = model.predict(scaled_features)[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return str(prediction), elapsed_ms


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


def predict_right_hand_gesture_and_latency(
    right_hand_landmarks,
    scaler,
    svm_model,
    knn_model,
) -> tuple[Optional[str], Optional[float], bool, Optional[float], Optional[float]]:
    """Classify the right hand with SVM and measure SVM/KNN inference latency."""
    if right_hand_landmarks is None:
        return None, None, False, None, None

    features = extract_relative_features_from_landmarks(right_hand_landmarks)
    feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)

    _, svm_ms = time_model_prediction(svm_model, scaled_features)
    _, knn_ms = time_model_prediction(knn_model, scaled_features)

    svm_prediction, confidence = predict_with_confidence(svm_model, scaled_features)
    return svm_prediction, confidence, True, svm_ms, knn_ms


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


def map_delta_to_axis(delta: float) -> int:
    """Map a pixel delta to the XInput axis range with dead zone and square curve."""
    clamped_delta = clamp(delta, -SAVJ_R_MAX_PIXELS, SAVJ_R_MAX_PIXELS)
    if abs(clamped_delta) < SAVJ_DEAD_ZONE_PIXELS:
        return 0

    normalized = abs(clamped_delta) / SAVJ_R_MAX_PIXELS
    axis_value = math.copysign((normalized**2) * XINPUT_AXIS_MAX, clamped_delta)
    return int(clamp(axis_value, -XINPUT_AXIS_MAX, XINPUT_AXIS_MAX))


def send_left_joystick(axis_x: int, axis_y: int) -> None:
    """Send the left-stick axis values through vgamepad."""
    try:
        gamepad.left_joystick_float(
            x_value_float=axis_x / XINPUT_AXIS_MAX,
            y_value_float=axis_y / XINPUT_AXIS_MAX,
        )
        gamepad.update()
    except Exception as exc:
        print(f"[Warning] Failed to update left joystick: {exc}")


def extract_left_arm_vector(pose_landmarks, frame_width: int, frame_height: int) -> Optional[tuple[float, float]]:
    """Return the controller-left wrist minus shoulder vector in pixel space."""
    if pose_landmarks is None:
        return None

    landmarks = pose_landmarks.landmark
    left_shoulder = landmarks[CONTROL_LEFT_SHOULDER_LANDMARK]
    left_wrist = landmarks[CONTROL_LEFT_WRIST_LANDMARK]

    shoulder_x = left_shoulder.x * frame_width
    shoulder_y = left_shoulder.y * frame_height
    wrist_x = left_wrist.x * frame_width
    wrist_y = left_wrist.y * frame_height
    return wrist_x - shoulder_x, wrist_y - shoulder_y


def update_savj(frame, pose_landmarks, savj_state: SavjState) -> tuple[int, int, bool]:
    """Update SAVJ calibration or send the current left-stick position."""
    frame_height, frame_width = frame.shape[:2]
    left_arm_vector = extract_left_arm_vector(pose_landmarks, frame_width, frame_height)

    if left_arm_vector is None:
        send_left_joystick(0, 0)
        return 0, 0, savj_state.is_calibrated

    vector_x, vector_y = left_arm_vector
    if not savj_state.is_calibrated:
        savj_state.add_sample(vector_x, vector_y)
        send_left_joystick(0, 0)
        return 0, 0, False

    delta_x = vector_x - float(savj_state.neutral_x)
    delta_y = vector_y - float(savj_state.neutral_y)
    mapped_delta_x = clamp(delta_x, -SAVJ_R_MAX_PIXELS, SAVJ_R_MAX_PIXELS)
    mapped_delta_y = clamp(-delta_y, -SAVJ_R_MAX_PIXELS, SAVJ_R_MAX_PIXELS)
    axis_x = map_delta_to_axis(mapped_delta_x)
    axis_y = map_delta_to_axis(mapped_delta_y)
    send_left_joystick(axis_x, axis_y)
    return axis_x, axis_y, True


def get_savj_anchor_points(pose_landmarks, frame_width: int, frame_height: int) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Return controller-left shoulder and wrist points in pixel coordinates."""
    if pose_landmarks is None:
        return None

    landmarks = pose_landmarks.landmark
    shoulder = landmarks[CONTROL_LEFT_SHOULDER_LANDMARK]
    wrist = landmarks[CONTROL_LEFT_WRIST_LANDMARK]
    shoulder_point = (int(shoulder.x * frame_width), int(shoulder.y * frame_height))
    wrist_point = (int(wrist.x * frame_width), int(wrist.y * frame_height))
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
    """Draw pose and controller-right hand landmarks from MediaPipe Holistic output."""
    draw_savj_visual(frame, results.pose_landmarks)

    controller_right_hand_landmarks = get_controller_right_hand_landmarks(results)
    if controller_right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            controller_right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            HAND_LANDMARK_STYLE,
            GREEN_HAND_CONNECTION_STYLE,
        )


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
    savj_state: SavjState,
    savj_ready: bool,
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

    if not savj_state.is_calibrated:
        left_text = f"Calibrating Left Arm... {savj_state.sample_count}/{CALIBRATION_FRAMES}"
        left_color = (0, 165, 255)
    elif savj_ready:
        left_text = f"Left Stick: X={axis_x} Y={axis_y}"
        left_color = (0, 255, 0)
    else:
        left_text = "Left Stick: No Pose"
        left_color = (0, 0, 255)

    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, left_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def write_latency_row(writer, timestamp: str, svm_ms: Optional[float], knn_ms: Optional[float]) -> None:
    """Write one frame of SVM/KNN latency data."""
    writer.writerow(
        [
            timestamp,
            "" if svm_ms is None else f"{svm_ms:.6f}",
            "" if knn_ms is None else f"{knn_ms:.6f}",
        ]
    )


def write_resource_row(writer, process: psutil.Process, timestamp: str) -> None:
    """Write current CPU and memory usage for this Python process."""
    cpu_percent = process.cpu_percent(interval=None)
    memory_mb = process.memory_info().rss / (1024 * 1024)
    writer.writerow([timestamp, f"{cpu_percent:.2f}", f"{memory_mb:.2f}"])


def main() -> None:
    """Run real-time Holistic inference and virtual gamepad control."""
    scaler, svm_model, knn_model = load_runtime_objects()
    process = psutil.Process()
    process.cpu_percent(interval=None)

    prediction_history: deque[str] = deque(maxlen=DEBOUNCE_FRAMES)
    current_gesture: Optional[str] = None
    pressed_button = None
    no_hand_frames = 0
    last_resource_log_time = 0.0
    savj_state = SavjState()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {CAMERA_INDEX}")

    print("Phase 3 dual-hand gamepad controller started. Press q to quit.")
    print(f"Latency log: {LATENCY_LOG_PATH}")
    print(f"Resource log: {RESOURCE_LOG_PATH}")

    try:
        with LATENCY_LOG_PATH.open("w", newline="", encoding="utf-8") as latency_file, RESOURCE_LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as resource_file, mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as holistic_detector:
            latency_writer = csv.writer(latency_file)
            resource_writer = csv.writer(resource_file)
            latency_writer.writerow(["timestamp", "svm_ms", "knn_ms"])
            resource_writer.writerow(["timestamp", "cpu_percent", "memory_mb"])

            while True:
                success, frame = cap.read()
                if not success:
                    print("[Warning] Failed to read a camera frame; skipping this frame.")
                    continue

                frame = cv2.flip(frame, 1)
                timestamp = current_timestamp()

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic_detector.process(image_rgb)
                    draw_holistic_landmarks(frame, results)

                    axis_x, axis_y, savj_ready = update_savj(frame, results.pose_landmarks, savj_state)
                    controller_right_hand_landmarks = get_controller_right_hand_landmarks(results)
                    raw_gesture, confidence, has_right_hand, svm_ms, knn_ms = predict_right_hand_gesture_and_latency(
                        right_hand_landmarks=controller_right_hand_landmarks,
                        scaler=scaler,
                        svm_model=svm_model,
                        knn_model=knn_model,
                    )
                except Exception as exc:
                    print(f"[Warning] Current frame prediction failed: {exc}")
                    axis_x, axis_y, savj_ready = 0, 0, savj_state.is_calibrated
                    raw_gesture, confidence, has_right_hand, svm_ms, knn_ms = None, None, False, None, None

                write_latency_row(latency_writer, timestamp, svm_ms, knn_ms)
                latency_file.flush()

                now = time.monotonic()
                if now - last_resource_log_time >= 1.0:
                    write_resource_row(resource_writer, process, timestamp)
                    resource_file.flush()
                    last_resource_log_time = now

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
                    savj_state=savj_state,
                    savj_ready=savj_ready,
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
