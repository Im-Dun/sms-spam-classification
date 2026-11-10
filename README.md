# SMS Spam Classification

Bài tập lớn học phần học máy: phân loại tin nhắn SMS thành hai lớp `ham` (bình thường) và `spam` (tin nhắn rác) bằng các phương pháp biểu diễn văn bản và thuật toán học máy truyền thống.

## 1. Cấu trúc repository

```text
sms-spam-classification/
├── data/                 # Đặt dataset tải từ UCI tại đây
├── models/               # Các mô hình đã huấn luyện (.joblib)
├── notebooks/            # Notebook EDA
├── report/               # Báo cáo EDA và hình EDA
├── results/              # Metrics, confusion matrix, lỗi dự đoán, artifacts
├── src/
│   ├── data_loader.py    # Đọc và kiểm tra dữ liệu
│   ├── preprocessing.py  # Chuẩn hóa, tokenization và pipeline
│   ├── train.py          # Chia dữ liệu và huấn luyện mô hình
│   └── evaluate.py       # Đánh giá và phân tích lỗi
├── .gitignore
├── AI_USAGE.md
└── requirements.txt
```

## 2. Yêu cầu môi trường

- Python 3.10 trở lên.
- pip.
- Dataset SMS Spam Collection từ UCI.

## 3. Tải và đặt dữ liệu

Nguồn: [UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Tải và giải nén dataset, sau đó đặt tệp `SMSSpamCollection` trực tiếp vào thư mục `data/`:

```text
data/SMSSpamCollection
```

Dataset không được lưu trong repository. Xem `data/README.md` để biết hướng dẫn chi tiết.

## 4. Cài đặt

Chạy từ thư mục gốc repository:

```bash
python -m venv .venv
```

Kích hoạt môi trường:

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Chạy chương trình

Tất cả lệnh dưới đây được chạy từ thư mục gốc repository.

### 5.1. Khám phá dữ liệu (EDA)

Mở `notebooks/eda.ipynb` bằng Jupyter Notebook hoặc VS Code, sau đó chạy lần lượt các cell. Đảm bảo dataset đã được đặt đúng đường dẫn.

### 5.2. Huấn luyện

```bash
python -m src.train
```

Chương trình:
1. Đọc và kiểm tra dữ liệu.
2. Loại bỏ bản ghi trùng hoàn toàn theo cặp nhãn/nội dung.
3. Chia dữ liệu train/test theo tỷ lệ 80/20, stratify theo nhãn, `random_state=42`.
4. Huấn luyện bốn pipeline.
5. Lưu mô hình vào `models/`.
6. Lưu dự đoán, nhãn và nội dung test vào `results/predictions_and_test.joblib`, cùng cấu hình và số liệu huấn luyện vào `results/training_summary.json`.

### 5.3. Đánh giá

Sau khi huấn luyện hoàn tất:

```bash
python -m src.evaluate
```

Chương trình sử dụng predictions đã lưu, không huấn luyện lại. Kết quả gồm:
- `results/metrics.csv`: Accuracy, Precision, Recall, F1-score và TN/FP/FN/TP.
- `results/evaluation_summary.json`: kết quả dạng JSON.
- `results/confusion_matrix_*.png`: confusion matrix từng mô hình.
- `results/misclassified_*.csv`: các mẫu dự đoán sai, có phân loại FP/FN.
- `results/f1_comparison.png`: biểu đồ so sánh F1-score.

Trong đánh giá nhị phân, `spam` được chọn là positive class.

## 6. Các phương án thực nghiệm

| Phương án | Biểu diễn đặc trưng | Thuật toán |
|---|---|---|
| Baseline | Bag-of-Words | Multinomial Naive Bayes |
| 2 | TF-IDF | Multinomial Naive Bayes |
| 3 | TF-IDF | Logistic Regression |
| 4 | TF-IDF | LinearSVC |

Các pipeline BoW và TF-IDF dùng cùng bước chuẩn hóa/tokenization, unigram và bigram. Vectorizer được fit bên trong pipeline trên tập train để hạn chế data leakage.

## 7. Kết quả thực nghiệm

Kết quả đánh giá trên tập test gồm 1.034 mẫu, sau khi thống nhất cấu hình tiền xử lý cho cả bốn pipeline:

| Mô hình | Accuracy | Precision (spam) | Recall (spam) | F1-score (spam) |
|---|---:|---:|---:|---:|
| BoW + MultinomialNB | 98.94% | 99.18% | 92.37% | 95.65% |
| TF-IDF + MultinomialNB | 93.71% | 100.00% | 50.38% | 67.01% |
| TF-IDF + Logistic Regression | 96.33% | 98.95% | 71.76% | 83.19% |
| TF-IDF + LinearSVC | 98.55% | 97.54% | 90.84% | 94.07% |

Trong bốn phương án, BoW kết hợp Multinomial Naive Bayes đạt F1-score cao nhất (95.65%), tiếp theo là TF-IDF kết hợp LinearSVC (94.07%).

**Lưu ý**: Các kết quả được tạo từ lần chạy thực nghiệm mới nhất bằng src.train và src.evaluate. Nhãn spam được chọn làm positive class.

## 8. Khai báo sử dụng AI

Nhóm khai báo việc sử dụng công cụ AI trong `AI_USAGE.md`. Các thành viên cần kiểm tra, hiểu và chịu trách nhiệm về mã nguồn cũng như kết quả được nộp.

## 9. Ghi chú về kết quả và file sinh ra

Các mô hình và file kết quả trong `models/` và `results/` có thể được tạo lại bằng hai lệnh huấn luyện và đánh giá ở trên. Không commit môi trường ảo, cache Python hoặc dataset gốc.
