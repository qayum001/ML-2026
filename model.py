import os
import logging
import argparse

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
    def train(self, dataset_filename: str):
        LOGGER.info(f"TRAIN called with dataset={dataset_filename}")

    def predict(self, dataset_filename: str):
        LOGGER.info(f"PREDICT called with dataset={dataset_filename}")

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