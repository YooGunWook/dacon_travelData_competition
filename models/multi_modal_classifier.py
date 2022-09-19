import torch
from torch import nn
from torch.nn import functional as F


class MultiModalClassifier(nn):
    def __init__(self, nlp_model, cv_model, config):
        self.nlp_model = nlp_model
        self.cv_model = cv_model
        self.layer = nn.Linear(256, config["num_classes"])

    def forward(inputs):
        return
