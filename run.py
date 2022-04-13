import pytorch_lightning as pl
from argparse import ArgumentParser


def argLoader():
    parser = ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="train the model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = argLoader()
    if config.do_train:
        data = 
