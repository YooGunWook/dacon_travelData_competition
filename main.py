import os
import timm # torch image models (like huggingface library)
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

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
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    return
