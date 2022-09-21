import os
import json
import timm  # torch image models (like huggingface library)
import torch
import random
import numpy as np
import pandas as pd
import pickle
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


def test_model(model, test_dataloader, device=None):
    model.eval()
    pred_list = []

    for batch in test_dataloader:
        cv_batch = batch[0].to(device)
        nlp_inputs = batch[1].to(device)
        nlp_attentions = batch[2].to(device)
        batch_dict = {"input_ids": nlp_inputs, "attention_mask": nlp_attentions}
        with torch.no_grad():
            outputs = model(batch_dict, cv_batch)
        pred = torch.argmax(outputs).flatten().detach().cpu().numpy().tolist()
        pred_list += pred

    return pred_list


def main():
    seed_everything(3307)
    with open("./config/config.json", "r") as f:
        config = json.load(f)
    train = pd.read_csv("./data/train.csv")
    num_labels = len(train["cat3"].drop_duplicates().tolist())
    if "idx_to_label.json" not in os.listdir("./config"):
        id_to_label = {
            i: label for i, label in enumerate(train["cat3"].drop_duplicates().tolist())
        }
        with open("./config/idx_to_label.json", "w", encoding="utf-8") as f:
            json.dump(id_to_label, f, ensure_ascii=False)
        label_to_id = {
            label: i for i, label in enumerate(train["cat3"].drop_duplicates().tolist())
        }
        with open("./config/label_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(label_to_id, f, ensure_ascii=False)
    else:
        with open("./config/idx_to_label.json", "r", encoding="utf-8") as f:
            id_to_label = json.load(f)
        with open("./config/label_to_idx.json", "r", encoding="utf-8") as f:
            label_to_id = json.load(f)

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
    if "test_dataset.pkl" not in os.listdir("./data"):
        with open("./data/train_dataset.pkl", "wb") as f:
            pickle.dump(train_dataset, f)
        with open("./data/val_dataset.pkl", "wb") as f:
            pickle.dump(val_dataset, f)
        with open("./data/test_dataset.pkl", "wb") as f:
            pickle.dump(test_dataset, f)
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=config["batch_size"], shuffle=True
    # )
    # valid_dataloader = DataLoader(
    #     val_dataset, batch_size=config["batch_size"], shuffle=False
    # )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=config["batch_size"], shuffle=False
    # )
    # train_model = trainer.Trainer(
    #     model, train_dataloader, valid_dataloader, config, device
    # )
    # train_model.build_model()
    # train_model.train()

    # model.load_state_dict(torch.load("./model/model_res.pt"))
    # model.to(device)
    # test_res = test_model(model, test_dataloader, device)
    # res_data = pd.read_csv("./data/sample_submission.csv")
    # res_data["cat3"] = test_res
    # res_data.to_csv("res.csv", index=False)


if __name__ == "__main__":
    main()
