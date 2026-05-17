"""
Phase 1: static hand gesture feature extraction.

Run:
    python extract_features.py

This script reads the gesture image folders, detects one hand with MediaPipe
Hands, extracts 60 scale-normalized relative landmark features, and writes
them to gesture_dataset.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "gesture_data_sample"
OUTPUT_CSV = ROOT_DIR / "gesture_dataset.csv"

GESTURE_LABELS = {
    "clasp": "clasp",
    "down": "down",
    "home": "home",
    "lb": "lb",
    "left": "left",
    "leftrunning": "leftrunning",
    "lt": "lt",
    "paper": "paper",
    "rb": "rb",
    "right": "right",
    "rock": "rock",
    "rt": "rt",
    "scissors": "scissors",
    "thumb": "thumb",
    "up": "up",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(file_path: Path) -> bool:
    """Return True when the path points to a supported image file."""
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS


def extract_relative_features(
    image_path: Path,
    hands_detector: mp.solutions.hands.Hands,
) -> Optional[list[float]]:
    """
    Extract 60 relative hand-landmark features from a single image.

    Landmark 0 is used as the origin, and the distance from Landmark 0 to
    Landmark 9 is used as the scale reference.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[Skip] Cannot read image: {image_path}")
        return None

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(image_rgb)
    except Exception as exc:
        print(f"[Skip] Failed to process image: {image_path}. Reason: {exc}")
        return None

    if not result.multi_hand_landmarks:
        print(f"[Skip] No hand detected: {image_path}")
        return None

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
        features.extend(
            [
                (landmark.x - wrist.x) / base_distance,
                (landmark.y - wrist.y) / base_distance,
                (landmark.z - wrist.z) / base_distance,
            ]
        )

    if len(features) != 60:
        print(f"[Skip] Invalid feature length for {image_path}: {len(features)}")
        return None

    return features


def iter_gesture_images() -> list[tuple[Path, str]]:
    """Collect image paths from the supported gesture folders."""
    image_items: list[tuple[Path, str]] = []

    for folder_name, label in GESTURE_LABELS.items():
        gesture_dir = DATASET_DIR / folder_name
        if not gesture_dir.exists():
            print(f"[Warning] Gesture folder does not exist and was skipped: {gesture_dir}")
            continue

        for image_path in sorted(gesture_dir.rglob("*")):
            if is_image_file(image_path):
                image_items.append((image_path, label))

    return image_items


def write_dataset(rows: list[list[float | str]]) -> None:
    """Write extracted features to gesture_dataset.csv."""
    feature_columns = [f"feature_{index}" for index in range(1, 61)]
    header = feature_columns + ["label"]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """Run feature extraction for all supported gesture folders."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATASET_DIR}")

    image_items = iter_gesture_images()
    if not image_items:
        raise RuntimeError(f"No valid images found under: {DATASET_DIR}")

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
        raise RuntimeError("No valid feature rows were generated.")

    write_dataset(rows)

    print("=" * 60)
    print("Phase 1 feature extraction completed")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Successful samples: {len(rows)}")
    print(f"Skipped images: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
