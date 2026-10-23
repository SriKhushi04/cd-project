import pandas as pd

REQUIRED = {"user_id", "text", "timestamp", "community"}

def load_records(path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["community"] = df["community"].fillna("").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[(df.text != "") & (df.community != "") & df.timestamp.notna()]
    return df.drop_duplicates(["user_id", "text", "timestamp", "community"]).reset_index(drop=True)

def temporal_split(df, train_before, test_from, test_until):
    train_cut = pd.Timestamp(train_before, tz="UTC")
    test_start = pd.Timestamp(test_from, tz="UTC")
    test_end = pd.Timestamp(test_until, tz="UTC")
    if train_cut > test_start:
        raise ValueError("train-before must be <= test-from")
    train = df[df.timestamp < train_cut].copy()
    test = df[(df.timestamp >= test_start) & (df.timestamp < test_end)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty split: train={len(train)}, test={len(test)}; check dates.")
    return train.reset_index(drop=True), test.reset_index(drop=True)
