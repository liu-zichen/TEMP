from .taxo import TaxStruct
from .models import TEMPBert, TEMPElectra, TEMPAlbert, TEMPRoberta, TEMPXLNet, TEMPXLM
from .trainer import Trainer
from .sampler import Sampler, Dataset
from .eval import Eval
import os


def init_folder():
    needed_folders = [
        "./data",
        "./data/train/",
        "./data/log",
        "./data/eval",
        "./data/models",
        "./data/result",
        "./cache"
    ]
    for path in needed_folders:
        if not os.path.exists(path):
            os.mkdir(path)


init_folder()
