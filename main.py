from audioop import mul
import os
import json
import timm  # torch image models (like huggingface library)
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
    with open("./config/config.json", "r") as f:
        config = json.load(f)
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    nlp_m = nlp_model.NLPModel(AutoModel.from_pretrained(config["nlp_model"]), config)
    tokenizer = AutoTokenizer.from_pretrained(config["nlp_model"])
    cv_m = cv_model.CVModel(
        timm.create_model(config["cv_model"], pretrained=True), config
    )
    model = multi_modal_classifier.MultiModalClassifier(nlp_m, cv_m, config)

    return
