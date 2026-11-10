# EDA notebook

Notebook `eda.ipynb` thực hiện khám phá bộ dữ liệu SMS Spam Collection:
- Kiểm tra cấu trúc và số lượng bản ghi.
- Kiểm tra nhãn, dữ liệu thiếu và dữ liệu trùng.
- Phân tích phân phối `ham`/`spam`.
- Khảo sát độ dài tin nhắn và trực quan hóa.

Trước khi chạy, tải dataset theo hướng dẫn tại `data/README.md` và đặt tại `data/SMSSpamCollection`. Chạy các cell từ đầu đến cuối trong môi trường đã cài `requirements.txt`.

Không fit vectorizer trên toàn bộ dữ liệu trong notebook EDA; bước vector hóa phục vụ huấn luyện được thực hiện trong pipeline sau khi chia train/test.
