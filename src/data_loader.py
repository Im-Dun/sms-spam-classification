"""Load the SMS Spam Collection dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"


def find_dataset_file(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Find the original SMSSpamCollection file in the data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {data_dir}. "
            "Hãy đặt dataset đã giải nén vào data/SMSPSamCollection/."
        )

    preferred_names = {"smsspamcollection", "smsspamcollection.txt"}
    candidates = [p for p in data_dir.rglob("*") if p.is_file()]
    for path in candidates:
        if path.name.lower() in preferred_names:
            return path

    text_candidates = [
        p for p in candidates
        if p.suffix.lower() in {".txt", ".tsv", ".csv"}
        and "readme" not in p.name.lower()
    ]
    if len(text_candidates) == 1:
        return text_candidates[0]

    candidate_list = "\n".join(str(p.relative_to(data_dir)) for p in text_candidates[:20])
    raise FileNotFoundError(
        "Không xác định được file dataset. Hãy kiểm tra thư mục data/. "
        f"Các file văn bản tìm được:\n{candidate_list or '(không có)'}"
    )


def load_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Read the tab-separated dataset and return columns label, text."""
    path = find_dataset_file(data_dir)
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8",
            on_bad_lines="warn",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="latin-1",
            on_bad_lines="warn",
        )

    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"Đã đọc {path} nhưng không tìm thấy dữ liệu có nhãn ham/spam. "
            "Kiểm tra định dạng file và dấu phân cách tab."
        )
    return df


if __name__ == "__main__":
    data = load_dataset()
    print(f"Số dòng hợp lệ: {len(data):,}")
    print("\nPhân bố nhãn:")
    print(data["label"].value_counts())
    print(f"\nFile dữ liệu: {find_dataset_file()}")
