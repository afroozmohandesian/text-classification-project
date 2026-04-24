# simple_main.py
import json
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_json_any(path: Path) -> pd.DataFrame:
    """Loads array-JSON or NDJSON into a DataFrame."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)           # array JSON
        return pd.DataFrame(data)
    except Exception:
        return pd.read_json(path, lines=True)  # NDJSON fallback


def build_text(df: pd.DataFrame) -> pd.Series:
    """title + abstract + categories (very simple)."""
    def _combine(r):
        title = (r.get("title") or "")
        abstract = (r.get("abstract") or "")
        cats = r.get("categories")
        if isinstance(cats, list):
            cats = " ".join(cats)
        cats = cats or ""
        return f"{title}\n{abstract}\n{cats}"
    return df.apply(_combine, axis=1).fillna("")


def main():
    JSON_PATH = Path("sample_data.json")
    CSV_PATH = Path("sample_targets.csv")
    OUT_PRED = Path("predictions.csv")

    # 1) Load & merge
    df_json = load_json_any(JSON_PATH)
    df_tgt = pd.read_csv(CSV_PATH)         # expects: id,target
    df = df_tgt.merge(df_json, on="id", how="left")

    # 2) Text + target
    texts = build_text(df)
    y = df["target"].astype(int).values

    # 3) Quick validation split (80/20)
    X_tr, X_va, y_tr, y_va = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Minimal TF-IDF (fast & simple)
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 1))
    Xtr = vec.fit_transform(X_tr)
    Xva = vec.transform(X_va)

    # 5) Minimal classifier (OvR LR, liblinear is fast for small-ish features)
    clf = LogisticRegression(max_iter=500, multi_class="ovr", solver="liblinear")
    clf.fit(Xtr, y_tr)

    # 6) Quick metrics
    y_pred = clf.predict(Xva)
    print("=== Quick validation (20%) ===")
    print(f"Accuracy : {accuracy_score(y_va, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_va, y_pred, average='macro'):.4f}")
    print(f"F1-micro: {f1_score(y_va, y_pred, average='micro'):.4f}")
    print("\nReport:\n", classification_report(y_va, y_pred))

    # 7) Retrain on ALL data and write final probabilities
    vec_full = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 1))
    X_all = vec_full.fit_transform(texts)
    clf_full = LogisticRegression(max_iter=500, multi_class="ovr", solver="liblinear")
    clf_full.fit(X_all, y)

    proba = clf_full.predict_proba(X_all)          # (N, n_classes)
    classes = clf_full.classes_                    # e.g., [1,2,3,4]
    cols = [f"prob_class_{int(c)}" for c in classes]

    out = pd.DataFrame(proba, columns=cols)
    out.insert(0, "id", df["id"].values)
    out.to_csv(OUT_PRED, index=False)

    print(f"\n Saved predictions to: {OUT_PRED}")


if __name__ == "__main__":
    main()
