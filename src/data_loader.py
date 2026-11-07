import csv
from pathlib import Path

import pandas as pd


def load_dataset(file_path="data/SMSSpamCollection"):
    """
    Đọc và chuẩn hóa dataset SMS Spam Collection.
    Đảm bảo trả về DataFrame gồm đúng 2 cột: label và text.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {path}")

    records = []
    malformed_record_count = 0
    with path.open(encoding="utf-8", newline="") as dataset_file:
        for row in csv.reader(dataset_file, delimiter="\t"):
            if len(row) != 2:
                malformed_record_count += 1
                continue
            records.append(row)

    df = pd.DataFrame(records, columns=["label", "text"], dtype="string")
    initial_count = len(df) + malformed_record_count

    df["label"] = df["label"].str.strip().str.lower()
    missing_label_mask = df["label"].isna() | df["label"].eq("")
    missing_text_mask = df["text"].isna() | df["text"].str.strip().eq("")
    missing_record_mask = missing_label_mask | missing_text_mask
    missing_label_count = int(missing_label_mask.sum())
    missing_text_count = int(missing_text_mask.sum())
    missing_record_count = int(missing_record_mask.sum())
    df = df.loc[~missing_record_mask].copy()

    valid_label_mask = df["label"].isin(["ham", "spam"])
    invalid_label_count = int((~valid_label_mask).sum())
    df = df.loc[valid_label_mask].copy()

    duplicate_count = int(df.duplicated(subset=["label", "text"]).sum())
    df = df.drop_duplicates(subset=["label", "text"], keep="first").reset_index(drop=True)

    conflicting_texts = df.groupby("text")["label"].nunique()
    conflicting_texts = conflicting_texts[conflicting_texts > 1]
    conflicting_text_count = int(len(conflicting_texts))
    conflicting_record_count = int(df["text"].isin(conflicting_texts.index).sum())

    print(f"Số bản ghi ban đầu: {initial_count}")
    print(f"Số bản ghi sai định dạng TSV: {malformed_record_count}")
    print(f"Bản ghi thiếu nhãn: {missing_label_count}")
    print(f"Bản ghi thiếu nội dung: {missing_text_count}")
    print(f"Bản ghi thiếu nhãn hoặc nội dung: {missing_record_count}")
    print(f"Số bản ghi có nhãn không hợp lệ: {invalid_label_count}")
    print(f"Số bản ghi trùng hoàn toàn đã loại: {duplicate_count}")
    print(
        "Số nội dung có nhãn xung đột: "
        f"{conflicting_text_count}; số bản ghi liên quan (được giữ lại): "
        f"{conflicting_record_count}"
    )
    print(f"Số bản ghi sau xử lý: {len(df)}")
    print(f"Các cột: {df.columns.tolist()}")
    print("Phân bố nhãn:\n", df["label"].value_counts())

    return df


if __name__ == "__main__":
    df = load_dataset()
    print(
        "Tỷ lệ phân bố nhãn (%):\n",
        (df["label"].value_counts(normalize=True) * 100).round(2),
    )
