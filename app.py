from pathlib import Path

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODELS = {
    "BoW + MultinomialNB": "bow_nb.joblib",
    "TF-IDF + MultinomialNB": "tfidf_nb.joblib",
    "TF-IDF + Logistic Regression": "tfidf_logreg.joblib",
    "TF-IDF + LinearSVC": "tfidf_svm.joblib",
}


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / MODELS[model_name]
    return joblib.load(model_path)


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered",
)

st.title("SMS Spam Classification")
st.caption("Ứng dụng phân loại tin nhắn SMS bằng Machine Learning")

st.markdown(
    """
    Nhập nội dung tin nhắn để hệ thống dự đoán đó là **Spam**
    (tin nhắn rác) hay **Ham** (tin nhắn bình thường).
    """
)

model_name = st.selectbox(
    "Chọn mô hình",
    options=list(MODELS.keys()),
    index=0,
)

message = st.text_area(
    "Nội dung tin nhắn",
    placeholder="Nhập nội dung SMS tại đây...",
    height=150,
)

if st.button("Phân loại tin nhắn", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Vui lòng nhập nội dung tin nhắn.")
    else:
        try:
            model = load_model(model_name)
            prediction = model.predict([message])[0]

            st.divider()
            st.subheader("Kết quả dự đoán")

            if prediction == "spam":
                st.error("SPAM — Tin nhắn có dấu hiệu là tin nhắn rác.")
            else:
                st.success("HAM — Tin nhắn bình thường.")

            st.caption(f"Mô hình sử dụng: {model_name}")

        except FileNotFoundError:
            st.error("Không tìm thấy file model. Hãy kiểm tra thư mục models.")
        except Exception as error:
            st.error(f"Đã xảy ra lỗi: {error}")