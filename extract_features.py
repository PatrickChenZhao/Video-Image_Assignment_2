"""
Phase 1: 静态手势图片特征提取脚本

运行方式：
    python extract_features.py

功能说明：
    1. 遍历 gesture_data_sample/ 下的 rock、paper、scissors 三类图片。
    2. 使用 MediaPipe Hands 检测单手 21 个关键点。
    3. 以手腕点 Landmark 0 为坐标原点，提取其余 20 个点的相对 3D 坐标。
    4. 将 60 维特征和标签写入根目录 gesture_dataset.csv。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp


# 项目根目录：本脚本所在目录
ROOT_DIR = Path(__file__).resolve().parent

# 数据集目录：用户说明中所有 dataset/ 均指向 gesture_data_sample/
DATASET_DIR = ROOT_DIR / "gesture_data_sample"

# 输出 CSV 文件路径
OUTPUT_CSV = ROOT_DIR / "gesture_dataset.csv"

# 只处理这三类有效手势文件夹，并统一输出为指定标签字符串
GESTURE_LABELS = {
    "rock": "Rock",
    "paper": "Paper",
    "scissors": "Scissors",
}

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(file_path: Path) -> bool:
    """判断文件是否为支持的图片格式。"""
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS


def extract_relative_features(
    image_path: Path,
    hands_detector: mp.solutions.hands.Hands,
) -> Optional[list[float]]:
    """
    从单张图片中提取 60 维手部相对坐标特征。

    返回：
        - list[float]：成功检测到手部时，返回 20 个关键点 * 3 维坐标。
        - None：图片无法读取、无法检测到手部或处理异常时返回 None。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[跳过] 无法读取图片：{image_path}")
        return None

    try:
        # MediaPipe 需要 RGB 图像，OpenCV 默认读取为 BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(image_rgb)
    except Exception as exc:
        print(f"[跳过] 处理图片出错：{image_path}，原因：{exc}")
        return None

    if not result.multi_hand_landmarks:
        print(f"[跳过] 未检测到手部：{image_path}")
        return None

    # 只取检测到的第一只手，保证每张图片输出固定 60 维特征
    landmarks = result.multi_hand_landmarks[0].landmark
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
        # 核心约束：以手腕点 Landmark 0 为原点，计算其余 20 个点的相对坐标
        features.extend(
            [
                (landmark.x - wrist.x) / base_distance,
                (landmark.y - wrist.y) / base_distance,
                (landmark.z - wrist.z) / base_distance,
            ]
        )

    if len(features) != 60:
        print(f"[跳过] 特征维度异常：{image_path}，当前维度：{len(features)}")
        return None

    return features


def iter_gesture_images() -> list[tuple[Path, str]]:
    """
    收集三类手势图片路径及其标签。

    gesture_data_sample/ 中如果存在 none 或其他额外目录，会被自动忽略。
    """
    image_items: list[tuple[Path, str]] = []

    for folder_name, label in GESTURE_LABELS.items():
        gesture_dir = DATASET_DIR / folder_name
        if not gesture_dir.exists():
            print(f"[警告] 类别文件夹不存在，已跳过：{gesture_dir}")
            continue

        for image_path in sorted(gesture_dir.rglob("*")):
            if is_image_file(image_path):
                image_items.append((image_path, label))

    return image_items


def write_dataset(rows: list[list[float | str]]) -> None:
    """将提取结果写入 gesture_dataset.csv。"""
    feature_columns = [f"feature_{index}" for index in range(1, 61)]
    header = feature_columns + ["label"]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """脚本入口。"""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"数据集目录不存在：{DATASET_DIR}")

    image_items = iter_gesture_images()
    if not image_items:
        raise RuntimeError(f"未找到任何有效图片，请检查目录：{DATASET_DIR}")

    rows: list[list[float | str]] = []
    skipped_count = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands_detector:
        for image_path, label in image_items:
            features = extract_relative_features(image_path, hands_detector)
            if features is None:
                skipped_count += 1
                continue

            rows.append(features + [label])

    if not rows:
        raise RuntimeError("所有图片均未成功提取特征，未生成有效数据集。")

    write_dataset(rows)

    print("=" * 60)
    print("Phase 1 特征提取完成")
    print(f"数据集目录：{DATASET_DIR}")
    print(f"输出文件：{OUTPUT_CSV}")
    print(f"成功样本数：{len(rows)}")
    print(f"跳过图片数：{skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
