import sys

from src.eval.main_eval import eval_model
from src.train.main_train import train_model


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_model(sys.argv[2])
    elif sys.argv[1] == "eval":
        eval_model(sys.argv[2])
    else:
        raise NotImplementedError
