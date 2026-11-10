import os
import time
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# Import hàm đọc dữ liệu từ data_loader và các thí nghiệm từ preprocessing
from src.data_loader import load_dataset
from src.preprocessing import build_experiments


def run_training_pipeline(save_models=True):
    """
    Huấn luyện, so sánh các mô hình SMS Spam Classification
    và xuất kết quả/predictions cho Thành viên 4 (Evaluation & Integration).
    """
    print("=== [STEP 1] Tải và chuẩn bị dữ liệu ===")
    df = load_dataset()
    
    # Lấy số liệu thực tế từ dataframe
    X = df['text']
    y = df['label']
    total_samples = len(df)
    print(f"Tổng số mẫu thực tế (sau drop duplicates): {total_samples}")

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"Số lượng tập Train: {len(X_train)} | Số lượng tập Test: {len(X_test)}")

    # Lấy danh sách các thí nghiệm có sẵn từ Thành viên 2
    experiments = build_experiments()
    
    # Thí nghiệm bổ sung: TF-IDF + LinearSVC (SVM) để tối ưu trên không gian thưa
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    experiments['tfidf_svm'] = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC(random_state=42))
    ])

    trained_models = {}
    predictions = {}
    metrics = {}

    print("\n=== [STEP 2] Huấn luyện và Đánh giá sơ bộ ===")
    for exp_id, pipeline in experiments.items():
        start_time = time.time()
        
        # Fit pipeline trên X_train (tránh data leakage)
        pipeline.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # Dự đoán trên X_test
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        trained_models[exp_id] = pipeline
        predictions[exp_id] = y_pred
        metrics[exp_id] = {
            "accuracy": round(float(acc), 4),
            "train_time_seconds": round(elapsed_time, 4)
        }
        
        print(f"- {exp_id:12s} | Accuracy: {acc * 100:.2f}% | Time: {elapsed_time:.4f}s")

    # Lưu mô hình và artifact nếu được yêu cầu
    if save_models:
        print("\n=== [STEP 3] Lưu trữ mô hình & Kết quả cho Thành viên 4 ===")
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # 1. Lưu các mô hình .joblib
        for exp_id, model in trained_models.items():
            model_path = os.path.join("models", f"{exp_id}.joblib")
            joblib.dump(model, model_path)

        # 2. Lưu predictions và X_test, y_test cho Thành viên 4
        artifacts = {
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions
        }
        joblib.dump(artifacts, os.path.join("results", "predictions_and_test.joblib"))

        # 3. Lưu tổng quan huấn luyện (JSON)
        summary = {
            "data_summary": {
                "total_samples": total_samples,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "random_state": 42,
                "test_size": 0.2
            },
            "experiments": metrics
        }
        with open(os.path.join("results", "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        print("Đã lưu các mô hình vào thư mục 'models/'")
        print("Đã xuất kết quả dự đoán và dữ liệu test vào 'results/predictions_and_test.joblib'")

    return metrics, {
        "models": trained_models,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions
    }


if __name__ == "__main__":
    run_training_pipeline(save_models=True)