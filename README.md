# SMS Spam Classification

Đồ án học phần Trí tuệ nhân tạo / Nhập môn học máy: phân loại tin nhắn SMS thành `ham` (không phải spam) và `spam`.

## Cấu trúc

```text
data/       # Đặt dataset đã giải nén tại đây
notebooks/  # Notebook EDA
src/        # Mã nguồn Python
results/    # Kết quả đánh giá và phân tích lỗi
report/     # Báo cáo LaTeX
```

## 1. Chuẩn bị dữ liệu

Đặt dataset đã giải nén vào:

`data/SMSPSamCollection/`

Mã nguồn tự tìm file `SMSSpamCollection` (file thường không có phần mở rộng). Dataset gốc thường gồm mỗi dòng một nhãn và nội dung SMS, phân tách bằng tab. Không commit dataset nếu giảng viên không yêu cầu; giữ hướng dẫn nguồn tải trong `data/README.md`.

## 2. Cài thư viện

Python 3.10 trở lên được khuyến nghị.

```bash
python -m pip install -r requirements.txt
```

## 3. Chạy pipeline

Từ thư mục gốc của repository:

```bash
python -m src.train
```

Chương trình tạo các file trong `results/`:
- `metrics.csv`: Accuracy, Precision, Recall, F1 cho từng thí nghiệm.
- `confusion_matrix_*.png`: confusion matrix.
- `misclassified_*.csv`: các SMS bị dự đoán sai để phân tích lỗi.

Các mô hình so sánh:
1. Bag-of-Words + Multinomial Naive Bayes (baseline).
2. TF-IDF + Multinomial Naive Bayes.
3. TF-IDF + Logistic Regression.

Dữ liệu được chia train/test có stratification. Vectorizer nằm trong `Pipeline`, vì vậy chỉ được fit trên tập train để tránh data leakage. Kết quả chưa được điền sẵn; hãy chạy thực nghiệm và báo cáo số liệu thực tế.

## 4. EDA

Mở `notebooks/eda.ipynb` bằng Jupyter hoặc VS Code. Nếu đường dẫn dữ liệu của bạn khác, sửa biến `DATA_DIR` trong notebook.

## 5. Báo cáo

Báo cáo LaTeX nằm trong `report/`. Hãy đưa số liệu thực nghiệm thật vào báo cáo sau khi chạy code. Không sử dụng số liệu minh họa như kết quả thật.

## 6. Khai báo sử dụng AI

Xem `AI_USAGE.md` và chỉnh sửa theo đúng phần việc AI thực sự hỗ trợ. Các thành viên cần kiểm tra, hiểu và tự chịu trách nhiệm với phần code/báo cáo mình nộp.
