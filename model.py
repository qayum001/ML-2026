import os
import logging
import argparse
from typing import List

import pandas as pd

LOG_PATH = "./data/log_file.log"

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

class My_Classifier_Model:
    @staticmethod
    def _ensure_dirs():
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./model", exist_ok=True)

    @staticmethod
    def _read_csv(dataset_filename: str) -> pd.DataFrame:
        if not os.path.exists(dataset_filename):
            raise FileNotFoundError(f"Dataset file not found: {dataset_filename}")

        df = pd.read_csv(dataset_filename, low_memory=False)
        return df

    @staticmethod
    def _log_df_info(df: pd.DataFrame, name: str, max_cols: int = 50):
        cols: List[str] = list(df.columns)
        preview_cols = cols[:max_cols]
        more = "" if len(cols) <= max_cols else f" (+{len(cols) - max_cols} more)"
        LOGGER.info(f"{name}: shape={df.shape}")
        LOGGER.info(f"{name}: columns={preview_cols}{more}")

    def train(self, dataset_filename: str):
        self._ensure_dirs()
        LOGGER.info(f"TRAIN started. dataset={dataset_filename}")

        try:
            df = self._read_csv(dataset_filename)
            self._log_df_info(df, "train_df")

            if "Status" not in df.columns:
                raise ValueError("Train dataset must contain 'Status' column.")

            LOGGER.info("TRAIN finished (validation-only step).")

        except Exception as e:
            LOGGER.exception(f"TRAIN failed: {e}")
            raise

    def predict(self, dataset_filename: str):
        self._ensure_dirs()
        LOGGER.info(f"PREDICT started. dataset={dataset_filename}")

        try:
            df = self._read_csv(dataset_filename)
            self._log_df_info(df, "predict_df")

            if "id" not in df.columns:
                raise ValueError("Prediction dataset must contain 'id' column.")

            LOGGER.info("PREDICT finished (validation-only step).")

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