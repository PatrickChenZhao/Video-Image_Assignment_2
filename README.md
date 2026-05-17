# Gesture-Controlled Virtual Xbox Controller

This project uses MediaPipe hand and pose tracking, an SVM hand gesture classifier, and `vgamepad` to turn camera input into a virtual Xbox 360 controller on Windows.

The pipeline has three main stages:

1. Extract normalized hand landmark features from gesture images.
2. Train and evaluate static gesture classifiers.
3. Run a live camera controller that maps gestures and left-arm movement to gamepad input.

## Requirements

- Windows
- Python 3.10 or 3.11
- A webcam
- ViGEmBus installed for `vgamepad`
- A virtual environment is recommended

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Project Structure

```text
.
|-- gesture_data_sample/          # Gesture image dataset
|-- demogame/                     # Optional demo game assets and script
|-- extract_features.py           # Phase 1: extract 60 landmark features per image
|-- train_model.py                # Phase 2: train SVM and KNN gesture classifiers
|-- main.py                       # Phase 3: live virtual controller runtime
|-- gesture_dataset.csv           # Generated feature dataset
|-- scaler.pkl                    # Saved StandardScaler
|-- svm_model.pkl                 # Saved SVM runtime model
|-- knn_model.pkl                 # Saved KNN comparison model
|-- best_model.pkl                # Best model from training
|-- model_metrics_comparison.csv  # Evaluation metrics output
`-- confusion_matrix_comparison.csv
```

## Gesture Dataset

The runtime expects these 15 gesture classes:

```text
clasp, down, home, left, paper, rb, right, rt, scissors, thumb, up, rock, lb, leftrunning, lt
```

Each class should have its own folder under `gesture_data_sample/`. The scripts intentionally ignore unsupported folders such as `none`.

## Phase 1: Extract Features

Run:

```powershell
python extract_features.py
```

This script reads supported image files from `gesture_data_sample/`, detects one hand with MediaPipe Hands, extracts 60 normalized relative landmark features, and writes them to `gesture_dataset.csv`.

Feature extraction uses:

- Landmark `0` as the hand origin.
- The distance from Landmark `0` to Landmark `9` as the scale reference.
- Landmarks `1` through `20`, each represented by relative `x`, `y`, and `z`.

## Phase 2: Train Models

Run:

```powershell
python train_model.py
```

This script trains:

- `SVM_RBF` with probability output for the live controller.
- `KNN_5` for comparison.

It saves:

- `scaler.pkl`
- `svm_model.pkl`
- `knn_model.pkl`
- `best_model.pkl`
- evaluation CSV files

The live runtime in `main.py` uses `svm_model.pkl` and `scaler.pkl`.

## Phase 3: Run the Live Controller

Run:

```powershell
python main.py
```

Press `q` in the camera window to quit.

The runtime uses MediaPipe Holistic to read:

- `results.right_hand_landmarks` for the physical right hand.
- `results.left_hand_landmarks` for the physical left hand.
- `pose_landmarks` Landmark `11` and Landmark `15` for the left-arm virtual joystick.

Camera mirroring compensation is not applied in the current implementation.

## Gamepad Mapping

### Right Hand

| Gesture | Xbox Input |
| --- | --- |
| `clasp` | `Y` |
| `down` | D-pad Down |
| `home` | Start |
| `left` | D-pad Left |
| `paper` | `A` |
| `rb` | Right Shoulder |
| `right` | D-pad Right |
| `rt` | Right Trigger axis, full press |
| `scissors` | `X` |
| `thumb` | `B` |
| `up` | D-pad Up |
| `rock` | Release all right-hand buttons and Right Trigger |

### Left Hand

| Gesture | Xbox Input |
| --- | --- |
| `lb` | Left Shoulder |
| `leftrunning` | Left Stick Click |
| `lt` | Left Trigger axis, full press |

If the left hand is detected but the smoothed gesture is not one of `lb`, `leftrunning`, or `lt`, the runtime releases Left Shoulder, Left Trigger, and Left Stick Click.

## Debouncing and UI State

`main.py` keeps independent queues for the left and right hands:

```python
left_gesture_queue = deque(maxlen=3)
right_gesture_queue = deque(maxlen=3)
```

Each hand uses majority voting before changing the active gamepad state. The on-screen status text also uses the debounced state, not the instant model prediction, so the UI is less likely to flicker between labels.

## Shoulder-Anchored Virtual Joystick

The left Xbox joystick is controlled by the physical left arm:

- Left Shoulder: MediaPipe Pose Landmark `11`
- Left Wrist: MediaPipe Pose Landmark `15`

The joystick uses a shoulder-anchored local coordinate system:

```text
rel_x = wrist_x - shoulder_x
rel_y = shoulder_y - wrist_y
```

It then separates vector direction from movement strength:

1. Compute distance with `math.hypot(rel_x, rel_y)`.
2. Apply a deadzone of `40` pixels.
3. Clamp active distance to a maximum radius of `150` pixels.
4. Apply a squared power curve.
5. Send the result through `gamepad.left_joystick(x_value=joy_x, y_value=joy_y)`.

The camera overlay draws a line between the shoulder and wrist and prints the current `joy_x` and `joy_y` values for debugging.

## Notes

- `vgamepad` requires Windows and a working virtual gamepad driver.
- If `main.py` reports missing or mismatched model labels, rerun `extract_features.py` and `train_model.py`.
- The generated model, CSV, and image metric files are included so the live controller can run without retraining first.
