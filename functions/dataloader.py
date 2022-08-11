from typing import Tuple
import h5py
import json
import numpy as np
import torch
from torch import Tensor
from torch.nn import ConstantPad1d
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


class HDF5Dataset(data.Dataset):

    def __init__(self,
                 hdf5_path: str,
                 captions_path: str,
                 lengthes_path: str,
                 pad_id: float,
                 transform=None):
        super().__init__()

        self.pad_id = pad_id

        with h5py.File(hdf5_path) as h5_file:
            self.images_nm, = h5_file.keys()
            self.images = np.array(h5_file[self.images_nm])

        with open(captions_path, 'r') as json_file:
            self.captions = json.load(json_file)

        with open(lengthes_path, 'r') as json_file:
            self.lengthes = json.load(json_file)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:

        X = torch.as_tensor(self.images[i], dtype=torch.float) / 255.
        if self.transform:
            X = self.transform(X)

        y = [torch.as_tensor(c, dtype=torch.long) for c in self.captions[i]]
        y = pad_sequence(y, padding_value=self.pad_id)  # type: Tensor

        ls = torch.as_tensor(self.lengthes[i], dtype=torch.long)


        return X, y, ls

    def __len__(self):
        return self.images.shape[0]


class collate_pad(object):

    def __init__(self, max_len, pad_id=0):
        self.max_len = max_len
        self.pad = pad_id

    def __call__(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Pads batch of variable lengths to a fixed length (max_len)
        """
        X, y, ls = zip(*batch)
        X: Tuple[Tensor]
        y: Tuple[Tensor]
        ls: Tuple[Tensor]

        ls = torch.stack(ls)  # (B, num_captions)
        y = pad_sequence(y, batch_first=True, padding_value=self.pad)

        # pad to the max len
        pad_right = self.max_len - y.size(1)
        if pad_right > 0:
            # [B, captns_num, max_seq_len]
            y = y.permute(0, 2, 1)  # type: Tensor
            y = ConstantPad1d((0, pad_right), value=self.pad)(y)
            y = y.permute(0, 2, 1)  # [B, max_len, captns_num]

        X = torch.stack(X)  # (B, 3, 256, 256)

        return X, y, ls
