import json
import os
import logging
import argparse
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from catboost import CatBoostClassifier

LOG_PATH = "./data/log_file.log"
DATA_DIR = "./data"
MODEL_DIR = os.path.join(DATA_DIR, "model")

def setup_logger() -> logging.Logger:
    os.makedirs("./data", exist_ok=True)

    logger = logging.getLogger("cirrhosis")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger

LOGGER = setup_logger()

@dataclass
class ArtifactsMeta:
    features: List[str]
    cat_features: List[str]
    classes: List[str]

class My_Classifier_Model:
    def __init__(self):
        self.params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": 1500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_seed": 42,
            "verbose": 200,
        }
        self.model: Optional[CatBoostClassifier] = None
        self.meta: Optional[ArtifactsMeta] = None

    @staticmethod
    def _ensure_dirs():
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

    @staticmethod
    def _read_csv(dataset_filename: str) -> pd.DataFrame:
        if not os.path.exists(dataset_filename):
            raise FileNotFoundError(f"Dataset file not found: {dataset_filename}")

        df = pd.read_csv(dataset_filename, low_memory=False)
        return df

    @staticmethod
    def _default_cat_features(df: pd.DataFrame) -> List[str]:
        candidates = ["Sex", "Drug", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
        return [c for c in candidates if c in df.columns]

    @staticmethod
    def _log_df_info(df: pd.DataFrame, name: str, max_cols: int = 50):
        cols: List[str] = list(df.columns)
        preview_cols = cols[:max_cols]
        more = "" if len(cols) <= max_cols else f" (+{len(cols) - max_cols} more)"
        LOGGER.info(f"{name}: shape={df.shape}")
        LOGGER.info(f"{name}: columns={preview_cols}{more}")

    def _load_artifacts(self):
        model_path = os.path.join(MODEL_DIR, "model.cbm")
        meta_path = os.path.join(MODEL_DIR, "meta.json")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Model artifacts not found. Run train first.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.meta = ArtifactsMeta(
            features=meta["features"],
            cat_features=meta["cat_features"],
            classes=meta["classes"],
        )

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    @staticmethod
    def _cast_cat_to_str(X: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
        X = X.copy()
        for c in cat_features:
            if c in X.columns:
                X[c] = X[c].astype("string").fillna("__MISSING__")
        return X

    def train(self, dataset_filename: str):
        self._ensure_dirs()
        LOGGER.info(f"TRAIN started. dataset={dataset_filename}")

        try:
            df = self._read_csv(dataset_filename)

            if "Status" not in df.columns:
                raise ValueError("Train dataset must contain 'Status' column.")

            y = df["Status"].astype(str)
            X = df.drop(columns=["Status"])

            if "id" in X.columns:
                X = X.drop(columns=["id"])

            features = list(X.columns)
            cat_features = self._default_cat_features(X)
            X = self._cast_cat_to_str(X, cat_features)

            classes = sorted(y.unique().tolist())
            self.meta = ArtifactsMeta(features=features, cat_features=cat_features, classes=classes)

            LOGGER.info(f"train_df: rows={len(df)} cols={df.shape[1]}")
            LOGGER.info(f"Features count={len(features)} | Cat features={cat_features}")
            LOGGER.info(f"Classes={classes}")

            self.model = CatBoostClassifier(**self.params)
            self.model.fit(X, y, cat_features=cat_features)

            model_path = os.path.join(MODEL_DIR, "model.cbm")
            meta_path = os.path.join(MODEL_DIR, "meta.json")

            self.model.save_model(model_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta.__dict__, f, ensure_ascii=False, indent=2)

            LOGGER.info(f"TRAIN finished. Saved model: {model_path}")
            LOGGER.info(f"TRAIN finished. Saved meta:  {meta_path}")

        except Exception as e:
            LOGGER.exception(f"TRAIN failed: {e}")
            raise

    def predict(self, dataset_filename: str):
        self._ensure_dirs()
        LOGGER.info(f"PREDICT started. dataset={dataset_filename}")

        try:
            self._load_artifacts()

            df = self._read_csv(dataset_filename)
            if "id" not in df.columns:
                raise ValueError("Prediction dataset must contain 'id' column.")

            ids = df["id"]
            X = df.drop(columns=["id"], errors="ignore")
            X = X.reindex(columns=self.meta.features)
            X = self._cast_cat_to_str(X, self.meta.cat_features)

            proba = self.model.predict_proba(X)

            out = pd.DataFrame({"id": ids})
            for j, cls in enumerate(self.meta.classes):
                out[f"Status_{cls}"] = proba[:, j]

            out_path = os.path.join(DATA_DIR, "results.csv")
            out.to_csv(out_path, index=False)

            LOGGER.info(f"PREDICT finished. Saved: {out_path}")

        except Exception as e:
            LOGGER.exception(f"PREDICT failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--dataset", required=True)

    args = parser.parse_args()
    model = My_Classifier_Model()

    if args.command == "train":
        model.train(args.dataset)
    else:
        model.predict(args.dataset)

if __name__ == "__main__":
    main()