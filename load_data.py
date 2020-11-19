import os
import torch
import argparse
import torchvision.datasets as dsets
import torchvision.transforms as transforms

if __name__ == "__main__" :

    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        '--data_path', type=str,
        help="input data path"
    )

    args = argument_parser.parse.args()
    print("data loading...")

    mnist_train = dsets.CIFAR10(root = args.data_path,
                                train = True,
                                download = True)
    mnist_test = dsets.CIFAR10(root = args.data_path,
                                train = False, 
                                download = True)
    print("load complete")
