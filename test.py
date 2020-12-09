# TODO: predict with the optimal checkpoint
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--gpu", type=int)

    return parser.parse_args()
