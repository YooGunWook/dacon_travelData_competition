import os
import json
from re import X
import timm  # torch image models (like huggingface library)
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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
    num_labels = len(train["cat3"].drop_duplicates().tolist())
    label_to_id = {
        label: i for i, label in enumerate(train["cat3"].drop_duplicates().tolist())
    }
    train["cat3"] = train["cat3"].apply(lambda x: label_to_id[x])
    config["num_classes"] = num_labels
    X_train, X_val, y_train, y_val = train_test_split(
        train.drop(columns=["cat3"]),
        train["cat3"],
        test_size=0.1,
        random_state=3307,
        stratify=train["cat3"],
    )
    device = torch.device("cpu")
    test = pd.read_csv("./data/test.csv")
    nlp_m = nlp_model.NLPModel(AutoModel.from_pretrained(config["nlp_model"]), config)
    tokenizer = AutoTokenizer.from_pretrained(config["nlp_model"])
    cv_m = cv_model.CVModel(
        timm.create_model(config["cv_model"], pretrained=True), config
    )
    cv_config = resolve_data_config({}, model=cv_m.model)
    transform = create_transform(**cv_config)

    model = multi_modal_classifier.MultiModalClassifier(nlp_m, cv_m, config)
    train_dataset = custom_dataset.CustomDataset(
        tokenizer,
        X_train["img_path"].tolist(),
        X_train["overview"].tolist(),
        transform,
        config,
        y_train.tolist(),
    )
    val_dataset = custom_dataset.CustomDataset(
        tokenizer,
        X_val["img_path"].tolist(),
        X_val["overview"].tolist(),
        transform,
        config,
        y_val.tolist(),
    )
    test_dataset = custom_dataset.CustomDataset(
        tokenizer,
        test["img_path"].tolist(),
        test["overview"].tolist(),
        transform,
        config,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    valid_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    train_model = trainer.Trainer(
        model, train_dataloader, valid_dataloader, config, device
    )
    train_model.build_model()
    train_model.train()


if __name__ == "__main__":
    main()
