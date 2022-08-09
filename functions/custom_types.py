from typing import TypeVar
import torch.utils.data
from typing import DefaultDict, List, Dict, Mapping
from collections import Counter

ModelType = TypeVar("ModelType", bound=torch.nn.Module)
OptimType = TypeVar("OptimType", bound=torch.optim.Optimizer)
SchedulerType = TypeVar("SchedulerType", bound=torch.optim.lr_scheduler.StepLR)
DeviceTye = TypeVar("DeviceTye", bound=torch.device)
DataIterType = TypeVar("DataIterType", bound=torch.utils.data.DataLoader)

"""
Definition of some custom data types:
    Captions = DefaultDict[str, List[List[str]]]:
        a three-level structured dictionary based on image-id,
            caption-->captions-->annotations
    ImagesAndCaptions = Dict[str, Captions]:
        a images captions dic data use the image-id as the keys

"""

Captions = DefaultDict[str, List[List[str]]]
ImagesAndCaptions = Dict[str, Captions]


class BOW(Counter, Mapping[str, int]):
    pass
