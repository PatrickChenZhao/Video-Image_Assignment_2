"""
Phase 3: 基于静态手势识别的实时游戏键盘控制器

运行方式：
    python main.py

新增日志：
    1. latency_log.csv: 每一帧记录 [timestamp, svm_ms, knn_ms]。
    2. resource_log.csv: 每秒记录当前 Python 进程 CPU 占用和内存消耗。
"""

from __future__ import annotations

import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import psutil
from pynput.keyboard import Controller, Key

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# =========================
# 可手动微调的全局常量
# =========================

# 防抖帧数：只有连续 N 帧预测结果一致，才确认手势有效
DEBOUNCE_FRAMES = 3

# 置信度阈值：只有最佳模型预测概率达到该值，才允许触发按键
# 如果某个手势总是能识别但不触发，可先调低到 0.50~0.70 之间排查。
CONFIDENCE_THRESHOLD = 0.60

# 连续未检测到手部的帧数达到该值后，释放所有游戏按键
NO_HAND_RELEASE_FRAMES = 3

# 手势到游戏动作和按键的映射
KEY_MAPPING = {
    "Rock": {
        "action": "SQUAT",
        "key": Key.ctrl,
        "display": "Rock: SQUAT",
    },
    "Scissors": {
        "action": "SHOOT",
        "key": "j",
        "display": "Scissors: SHOOT",
    },
    "Paper": {
        "action": "JUMP",
        "key": Key.space,
        "display": "Paper: JUMP",
    },
}

# 摄像头编号，通常内置或默认摄像头为 0
CAMERA_INDEX = 0

# MediaPipe 检测参数
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6


# =========================
# 路径配置
# =========================

ROOT_DIR = Path(__file__).resolve().parent
SCALER_PATH = ROOT_DIR / "scaler.pkl"
BEST_MODEL_PATH = ROOT_DIR / "best_model.pkl"
SVM_MODEL_PATH = ROOT_DIR / "svm_model.pkl"
KNN_MODEL_PATH = ROOT_DIR / "knn_model.pkl"
LATENCY_LOG_PATH = ROOT_DIR / "latency_log.csv"
RESOURCE_LOG_PATH = ROOT_DIR / "resource_log.csv"


def current_timestamp() -> str:
    """返回适合 CSV 记录的毫秒级时间戳。"""
    return datetime.now().isoformat(timespec="milliseconds")


def load_runtime_objects():
    """加载 Phase 2 保存的 scaler、最佳模型、SVM 模型和 KNN 模型。"""
    required_files = [
        SCALER_PATH,
        BEST_MODEL_PATH,
        SVM_MODEL_PATH,
        KNN_MODEL_PATH,
    ]
    missing_files = [file_path for file_path in required_files if not file_path.exists()]
    if missing_files:
        missing_text = "\n".join(str(file_path) for file_path in missing_files)
        raise FileNotFoundError(
            "缺少运行所需文件：\n"
            f"{missing_text}\n"
            "请先重新运行 Phase 2：python train_model.py"
        )

    scaler = joblib.load(SCALER_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    knn_model = joblib.load(KNN_MODEL_PATH)
    return scaler, best_model, svm_model, knn_model


def extract_relative_features_from_landmarks(hand_landmarks) -> list[float]:
    """
    从 MediaPipe 手部关键点中提取 60 维相对坐标特征。

    注意：
        这里必须与 Phase 1 保持完全一致：
        以 Landmark 0 手腕点为坐标原点，计算其余 20 个点的相对 x、y、z。
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
        raise ValueError(f"实时特征维度异常，应为 60，实际为 {len(features)}")

    return features


def time_model_prediction(model, scaled_features) -> tuple[str, float]:
    """执行一次模型预测，并返回预测结果和耗时毫秒数。"""
    start_time = time.perf_counter()
    prediction = model.predict(scaled_features)[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return str(prediction), elapsed_ms


def predict_with_confidence(model, scaled_features) -> tuple[str, float]:
    """使用最佳模型预测手势，并返回最大类别概率作为置信度。"""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "当前 best_model.pkl 不支持 predict_proba()。"
            "请重新运行 Phase 2：python train_model.py"
        )

    probabilities = model.predict_proba(scaled_features)[0]
    best_index = int(np.argmax(probabilities))
    prediction = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return prediction, confidence


def predict_gesture_and_latency(
    frame,
    hands_detector,
    scaler,
    best_model,
    svm_model,
    knn_model,
) -> tuple[Optional[str], Optional[float], bool, Optional[float], Optional[float]]:
    """
    对当前帧进行手势预测，并记录 SVM/KNN 单独推理耗时。

    返回：
        gesture: 最佳模型预测出的手势名称；未检测到手部时为 None
        confidence: 最佳模型预测置信度；未检测到手部时为 None
        has_hand: 当前帧是否检测到手部
        svm_ms: SVM predict 耗时，未检测到手部时为 None
        knn_ms: KNN predict 耗时，未检测到手部时为 None
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None, None, False, None, None

    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    hand_landmarks = result.multi_hand_landmarks[0]
    features = extract_relative_features_from_landmarks(hand_landmarks)

    feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled_features = scaler.transform(feature_array)

    _, svm_ms = time_model_prediction(svm_model, scaled_features)
    _, knn_ms = time_model_prediction(knn_model, scaled_features)

    # 键盘控制逻辑只使用 Phase 2 选出的最佳模型，并读取概率作为置信度
    best_prediction, confidence = predict_with_confidence(best_model, scaled_features)
    return best_prediction, confidence, True, svm_ms, knn_ms


def get_stable_gesture(prediction_history: deque[str]) -> Optional[str]:
    """当历史队列已满且所有预测一致时，返回稳定手势。"""
    if len(prediction_history) < DEBOUNCE_FRAMES:
        return None

    first_prediction = prediction_history[0]
    if all(prediction == first_prediction for prediction in prediction_history):
        return first_prediction

    return None


def release_pressed_key(keyboard: Controller, pressed_key) -> None:
    """释放当前已按下的按键。"""
    if pressed_key is None:
        return

    try:
        keyboard.release(pressed_key)
    except Exception as exc:
        print(f"[警告] 释放按键失败：{pressed_key}，原因：{exc}")


def press_gesture_key(
    keyboard: Controller,
    gesture: str,
    current_gesture: Optional[str],
    pressed_key,
) -> tuple[Optional[str], object]:
    """
    根据稳定手势按下对应按键。

    如果手势发生切换，会先释放旧按键，再按下新按键。
    """
    mapping = KEY_MAPPING.get(gesture)
    if mapping is None:
        release_pressed_key(keyboard, pressed_key)
        return None, None

    target_key = mapping["key"]

    if gesture == current_gesture and pressed_key == target_key:
        return current_gesture, pressed_key

    release_pressed_key(keyboard, pressed_key)

    try:
        keyboard.press(target_key)
        return gesture, target_key
    except Exception as exc:
        print(f"[警告] 按下按键失败：{target_key}，原因：{exc}")
        return None, None


def draw_status(
    frame,
    raw_gesture: Optional[str],
    confidence: Optional[float],
    stable_gesture: Optional[str],
    has_hand: bool,
) -> None:
    """在画面上绘制当前预测状态。"""
    if stable_gesture in KEY_MAPPING:
        confidence_text = "" if confidence is None else f" ({confidence * 100:.1f}%)"
        status_text = f"{KEY_MAPPING[stable_gesture]['display']}{confidence_text}"
        color = (0, 255, 0)
    elif raw_gesture in KEY_MAPPING and confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        status_text = f"Low Confidence: {raw_gesture} ({confidence * 100:.1f}%)"
        color = (0, 165, 255)
    elif raw_gesture in KEY_MAPPING:
        confidence_text = "" if confidence is None else f" ({confidence * 100:.1f}%)"
        status_text = f"Detecting: {raw_gesture}{confidence_text}"
        color = (0, 255, 255)
    elif not has_hand:
        status_text = "No Hand: RELEASE"
        color = (0, 0, 255)
    else:
        status_text = "Unknown: RELEASE"
        color = (0, 0, 255)

    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def write_latency_row(writer, timestamp: str, svm_ms: Optional[float], knn_ms: Optional[float]) -> None:
    """写入单帧 SVM/KNN 延迟记录。"""
    writer.writerow(
        [
            timestamp,
            "" if svm_ms is None else f"{svm_ms:.6f}",
            "" if knn_ms is None else f"{knn_ms:.6f}",
        ]
    )


def write_resource_row(writer, process: psutil.Process, timestamp: str) -> None:
    """写入当前 Python 进程 CPU 占用和内存消耗。"""
    cpu_percent = process.cpu_percent(interval=None)
    memory_mb = process.memory_info().rss / (1024 * 1024)
    writer.writerow([timestamp, f"{cpu_percent:.2f}", f"{memory_mb:.2f}"])


def main() -> None:
    """实时推理与键盘控制主程序入口。"""
    scaler, best_model, svm_model, knn_model = load_runtime_objects()
    keyboard = Controller()
    process = psutil.Process()
    process.cpu_percent(interval=None)

    prediction_history: deque[str] = deque(maxlen=DEBOUNCE_FRAMES)
    current_gesture: Optional[str] = None
    pressed_key = None
    no_hand_frames = 0
    last_resource_log_time = 0.0

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头，摄像头编号：{CAMERA_INDEX}")

    print("Phase 3 实时控制已启动。按 q 退出程序。")
    print(f"逐帧延迟日志：{LATENCY_LOG_PATH}")
    print(f"资源占用日志：{RESOURCE_LOG_PATH}")

    try:
        with LATENCY_LOG_PATH.open("w", newline="", encoding="utf-8") as latency_file, RESOURCE_LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as resource_file, mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands_detector:
            latency_writer = csv.writer(latency_file)
            resource_writer = csv.writer(resource_file)
            latency_writer.writerow(["timestamp", "svm_ms", "knn_ms"])
            resource_writer.writerow(["timestamp", "cpu_percent", "memory_mb"])

            while True:
                success, frame = cap.read()
                if not success:
                    print("[警告] 无法读取摄像头画面，已跳过当前帧。")
                    continue

                frame = cv2.flip(frame, 1)
                timestamp = current_timestamp()

                try:
                    raw_gesture, confidence, has_hand, svm_ms, knn_ms = predict_gesture_and_latency(
                        frame=frame,
                        hands_detector=hands_detector,
                        scaler=scaler,
                        best_model=best_model,
                        svm_model=svm_model,
                        knn_model=knn_model,
                    )
                except Exception as exc:
                    print(f"[警告] 当前帧预测失败：{exc}")
                    raw_gesture, confidence, has_hand, svm_ms, knn_ms = None, None, False, None, None

                write_latency_row(latency_writer, timestamp, svm_ms, knn_ms)
                latency_file.flush()

                now = time.monotonic()
                if now - last_resource_log_time >= 1.0:
                    write_resource_row(resource_writer, process, timestamp)
                    resource_file.flush()
                    last_resource_log_time = now

                stable_gesture = None

                is_confident = (
                    has_hand
                    and raw_gesture in KEY_MAPPING
                    and confidence is not None
                    and confidence >= CONFIDENCE_THRESHOLD
                )

                if is_confident:
                    no_hand_frames = 0
                    prediction_history.append(raw_gesture)
                    stable_gesture = get_stable_gesture(prediction_history)

                    if stable_gesture is not None:
                        current_gesture, pressed_key = press_gesture_key(
                            keyboard=keyboard,
                            gesture=stable_gesture,
                            current_gesture=current_gesture,
                            pressed_key=pressed_key,
                        )
                elif has_hand:
                    prediction_history.clear()
                    no_hand_frames = 0
                    release_pressed_key(keyboard, pressed_key)
                    current_gesture = None
                    pressed_key = None
                else:
                    prediction_history.clear()
                    no_hand_frames += 1

                    if no_hand_frames >= NO_HAND_RELEASE_FRAMES:
                        release_pressed_key(keyboard, pressed_key)
                        current_gesture = None
                        pressed_key = None

                draw_status(frame, raw_gesture, confidence, stable_gesture, has_hand)
                cv2.imshow("Static Gesture Game Controller", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出程序。")
    finally:
        release_pressed_key(keyboard, pressed_key)
        cap.release()
        cv2.destroyAllWindows()
        print("已释放摄像头、窗口和所有游戏按键。")


if __name__ == "__main__":
    main()
