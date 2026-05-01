"""
Phase 2: 静态手势识别模型训练与评估脚本

运行方式：
    python train_model.py

功能说明：
    1. 读取 Phase 1 生成的 gesture_dataset.csv。
    2. 将数据拆分为 80% 训练集和 20% 测试集。
    3. 使用 StandardScaler 对特征进行标准化。
    4. 分别训练 SVM 和 KNN 模型。
    5. 输出两个模型的 Accuracy 和 Confusion Matrix。
    6. 保存准确率最高的模型为 best_model.pkl，同时保存 scaler.pkl。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 项目根目录：本脚本所在目录
ROOT_DIR = Path(__file__).resolve().parent

# Phase 1 输出的数据集
DATASET_CSV = ROOT_DIR / "gesture_dataset.csv"

# Phase 2 输出的模型与标准化器
SCALER_PATH = ROOT_DIR / "scaler.pkl"
BEST_MODEL_PATH = ROOT_DIR / "best_model.pkl"
SVM_MODEL_PATH = ROOT_DIR / "svm_model.pkl"
KNN_MODEL_PATH = ROOT_DIR / "knn_model.pkl"

# 固定随机种子，保证每次划分结果可复现
RANDOM_STATE = 42

# 标签列名
LABEL_COLUMN = "label"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    读取 gesture_dataset.csv，并拆分特征 X 与标签 y。

    返回：
        X: 60 维特征数据
        y: 手势标签，包含 Rock、Paper、Scissors
    """
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"未找到数据集文件：{DATASET_CSV}\n"
            "请先运行 Phase 1：python extract_features.py"
        )

    data = pd.read_csv(DATASET_CSV)
    if data.empty:
        raise ValueError("gesture_dataset.csv 为空，无法训练模型。")

    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"数据集中缺少标签列：{LABEL_COLUMN}")

    feature_columns = [column for column in data.columns if column != LABEL_COLUMN]
    if len(feature_columns) != 60:
        raise ValueError(
            f"特征列数量应为 60，但当前为 {len(feature_columns)}。"
            "请检查 Phase 1 的特征提取结果。"
        )

    X = data[feature_columns]
    y = data[LABEL_COLUMN]

    if y.nunique() < 2:
        raise ValueError("标签类别少于 2 类，无法完成有效训练。")

    return X, y


def print_dataset_summary(y: pd.Series) -> None:
    """打印当前数据集的类别分布。"""
    print("=" * 60)
    print("数据集概览")
    print(f"总样本数：{len(y)}")
    print("类别分布：")
    for label, count in y.value_counts().sort_index().items():
        print(f"  {label}: {count}")
    print("=" * 60)


def evaluate_model(
    model_name: str,
    model,
    X_test,
    y_test,
    labels: list[str],
) -> float:
    """
    在测试集上评估模型，并打印 Accuracy 和 Confusion Matrix。

    返回：
        accuracy: 测试集准确率
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\n[{model_name}]")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    return accuracy


def main() -> None:
    """脚本入口。"""
    X, y = load_dataset()
    print_dataset_summary(y)

    # 使用 stratify 保持训练集和测试集中的类别比例接近原始数据
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 核心约束：必须使用 StandardScaler 对特征数据进行拟合和转换
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM_RBF": SVC(kernel="rbf", probability=True),
        "KNN_5": KNeighborsClassifier(n_neighbors=5),
    }

    trained_models = {}
    scores = {}
    labels = sorted(y.unique().tolist())

    for model_name, model in models.items():
        print(f"\n开始训练模型：{model_name}")
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        scores[model_name] = evaluate_model(
            model_name=model_name,
            model=model,
            X_test=X_test_scaled,
            y_test=y_test,
            labels=labels,
        )

    # 如果准确率相同，max 会保留字典插入顺序中先出现的模型，即优先 SVM_RBF
    best_model_name = max(scores, key=scores.get)
    best_model = trained_models[best_model_name]
    best_accuracy = scores[best_model_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(trained_models["SVM_RBF"], SVM_MODEL_PATH)
    joblib.dump(trained_models["KNN_5"], KNN_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n" + "=" * 60)
    print("Phase 2 模型训练完成")
    print(f"最佳模型：{best_model_name}")
    print(f"最佳 Accuracy：{best_accuracy:.4f}")
    print(f"模型已保存：{BEST_MODEL_PATH}")
    print(f"SVM 模型已保存：{SVM_MODEL_PATH}")
    print(f"KNN 模型已保存：{KNN_MODEL_PATH}")
    print(f"Scaler 已保存：{SCALER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
