"""
This file sed to hold some global, generalised data processing functions
    def parse_arguments():
        Setting various path parameters
    def load_json():
        Loading a json file
    def write_json:
        Saving data as json file
    def write_h5_dataset():
        Processing the data into hdf5 format
    def seed_worker():
        Setting a random seed
    def init_unk():
        Initialization unknown word vectors, taking in a Tensor and returns a weight Tensor of the same size
"""

import random
from typing import List, Tuple
from argparse import Namespace
from numpy.typing import NDArray

import argparse
import json
import h5py

import numpy as np
import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LMU-SS2022-CVDL-Project")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets-original/",
        help="Directory contains MS COCO dataset_functional files.")

    parser.add_argument(
        "--json_train",
        type=str,
        default="annotations_trainval2017/annotations/captions_train2017.json",
        help="Directory have MS COCO annotations file for the train split.")

    parser.add_argument(
        "--json_val",
        type=str,
        default="annotations_trainval2017/annotations/captions_val2017.json",
        help="Directory have MS COCO annotations file for the val split.")

    parser.add_argument(
        "--image_train",
        type=str,
        default="train2017",
        help="Directory have MS COCO images files for the train split.")

    parser.add_argument(
        "--image_val",
        type=str,
        default="val2017",
        help="Directory have MS COCO image files for the val split.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets-processed",
        help="Directory to save the output files.")

    parser.add_argument("--vector_dir",
                        type=str,
                        default="embeddings/",
                        help="Directory to embedding vector.")

    parser.add_argument("--vector_dim",
                        type=str,
                        default="300",
                        help="Vector dimension")

    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="minimum frequency needed to include a token in the vocabulary")

    parser.add_argument(
        "--max_len",
        type=int,
        default=52,
        help="minimum length for captions")

    args = parser.parse_args()

    return args


def load_json(json_path: str) -> Tuple[list, List[str]]:
    with open(json_path) as json_file:
        data = json.load(json_file)

    annotations = data["annotations"]
    images = data["images"]

    return annotations, images


def write_json(json_path: str, data) -> None:
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def write_h5_dataset(write_path: str, data: NDArray, name: str,
                     type: str) -> None:
    with h5py.File(write_path, "w") as h5f:
        h5f.create_dataset(name=name,
                           data=data,
                           shape=np.shape(data),
                           dtype=type)


def seed_worker(worker_id):
    # ref: https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_unk(tensor: Tensor) -> Tensor:
    weight_unk = torch.ones(tensor.size())
    return xavier_uniform_(weight_unk.view(1, -1)).view(-1)
