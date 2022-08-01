"""
Definition of some custom data types:
    Captions = DefaultDict[str, List[List[str]]]:
        a three-level structured dictionary based on image-id,
            caption-->captions-->annotations
    ImagesAndCaptions = Dict[str, Captions]:
        a images captions dic data use the image-id as the keys

"""

from typing import DefaultDict, List, Dict, Mapping
from collections import Counter

Captions = DefaultDict[str, List[List[str]]]
ImagesAndCaptions = Dict[str, Captions]


class BOW(Counter, Mapping[str, int]):
    pass
