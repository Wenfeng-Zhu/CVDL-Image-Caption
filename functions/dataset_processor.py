"""
This file is used to store some processing functions for the dataset
    def get_captions():
        Getting a captions dictionary based on image-id
    def combine_image_captions():
        Combining the image-id and captions
    def load_images():
        Loading and transposing the images to (C,H,W) which can be used for Tensor format
    def encode_captions():
        Encode captions text to the respective indices, return the encoded captions and their lengths
    def split_dataset():
        The original data set (training and validation sets) is divided into training, validation
        and test sets in the ratio of 70%, 15% and 15%.
    def build_vocab():
        Construction of Vocab classes from local word vector files(here we used Glove.6B as the vector file)
    def create_input_arrays():
        return images and encode captions and their lengths as the input array
    def run_create_arrays():
        Accept actual data and create input arrays
"""

from typing import List, Tuple
from numpy.typing import NDArray
from functions.custom_types import Captions, ImagesAndCaptions, BOW

from collections import defaultdict, Counter
from itertools import chain
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import re
import numpy as np
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split
import cv2
from utils import init_unk


def get_captions(annotations: list, max_len: int) -> Captions:

    captions_dict = defaultdict(list)
    for annot in annotations:
        captions = [
            s for s in re.split(r"(\W)", annot["caption"]) if s.strip()
        ]
        if len(captions) > (max_len - 2):
            captions = captions[:max_len - 2]

        captions = ["<sos>"] + captions + ["<eos>"]
        captions_dict[annot["image_id"]].append(captions)

    return captions_dict


def combine_image_captions(images: List[str], captions_dict: Captions, images_dir: str) -> ImagesAndCaptions:
    images_w_captions = {}
    for img in images:
        img_id = img["id"]

        # select random 5 captions
        captions = captions_dict[img_id]
        idxs = np.random.choice(range(len(captions)), size=5, replace=False)
        captions_selected = [captions[i] for i in idxs]

        img_filename = images_dir + "/" + img["file_name"]
        images_w_captions[img_filename] = {
            "image_id": img_id,
            "captions": captions_selected
        }

    return images_w_captions


def load_images(image_path: str,
                resize_h: int = None,
                resize_w: int = None) -> NDArray:
    img = cv2.resize(cv2.imread(image_path), (resize_h, resize_w),
                     interpolation=cv2.INTER_AREA)  # type: NDArray
    return img.transpose(2, 0, 1)


def encode_captions(captions: List[List[str]],
                    vocab: Vocab) -> Tuple[List[List[int]], List[int]]:
    encoded = []
    lengths = []
    for caption in captions:
        encoded.append([vocab.stoi[s] for s in caption])
        lengths.append(len(caption))

    return encoded, lengths


def split_dataset(
        original_train_split: ImagesAndCaptions,
        original_val_split: ImagesAndCaptions,
        SEED: int,
        test_perc: int = 0.15,
        val_perc: int = 0.15
) -> Tuple[ImagesAndCaptions, ImagesAndCaptions, ImagesAndCaptions]:
    train_perc = 1 - (test_perc + val_perc)
    original_val_size = len(original_val_split)
    original_train_size = len(original_train_split)
    ds_size = original_val_size + original_train_size
    test_makeup_size = int(ds_size * val_perc) - original_val_size
    train_size = int((train_perc / (1 - test_perc)) *
                     (ds_size - original_val_size - test_makeup_size))
    original_train_list = list(original_train_split.items())
    test_makeup, train_val = train_test_split(original_train_list,
                                              train_size=test_makeup_size,
                                              random_state=SEED,
                                              shuffle=True)
    test_split = {**dict(test_makeup), **original_val_split}
    train_split, val_split = train_test_split(train_val,
                                              train_size=train_size,
                                              random_state=SEED,
                                              shuffle=True)

    return dict(train_split), dict(val_split), test_split


def build_vocab(captions: List[chain],
                vector_dir: str,
                vector_name: str,
                min_freq: int = 2) -> Vocab:
    all_words = list(chain.from_iterable(captions))  # Type: List[str]
    bag_of_words: BOW = Counter(all_words)

    vocab: Vocab = Vocab(bag_of_words,
                         min_freq=min_freq,
                         specials=("<unk>", "<pad>", "<sos>", "<eos>"),
                         vectors_cache=vector_dir,
                         vectors=vector_name,
                         unk_init=init_unk)
    # first_vocab = build_vocab_from_iterator(all_words, min_freq=min_freq, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
    # myvec = GloVe(name='6B', dim=300, )
    # myvocab = vocab(myvec.stoi)
    # myvocab.insert_token('<unk>', 0)
    # myvocab.insert_token('<pad>', 1)
    # myvocab.insert_token('<sos>', 2)
    # myvocab.insert_token('<eos>', 3)
    return vocab


def create_input_arrays(
        dataset: Tuple[str, Captions],
        vocab: Vocab) -> Tuple[NDArray, List[List[int]], List[int]]:
    image = load_images(dataset[0], 256, 256)
    captions_encoded, lengths = encode_captions(dataset[1]["captions"], vocab)

    return image, captions_encoded, lengths


def run_create_arrays(
        dataset: ImagesAndCaptions,
        vocab: Vocab,
        split: str,
) -> Tuple[NDArray, List[List[List[int]]], List[List[int]]]:
    f = partial(create_input_arrays, vocab=vocab)
    num_proc = mp.cpu_count()
    with mp.Pool(processes=num_proc) as pool:
        arrays = list(
            tqdm(pool.imap(f, dataset.items()),
                 total=len(dataset),
                 desc=f"Preparing {split} Dataset",
                 unit="Image"))

    images, captions_encoded, lengths = zip(*arrays)

    return np.stack(images), captions_encoded, lengths
