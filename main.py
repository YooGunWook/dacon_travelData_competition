import os
import torch
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer

from modules import custom_dataset, trainer
from models import cv_model, nlp_model, multi_modal_classifier


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    seed_everything(3307)
    return
