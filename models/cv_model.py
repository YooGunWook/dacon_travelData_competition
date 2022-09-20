import torch
from torch import nn
from torch.nn import functional as F


class CVModel(nn):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.encoder_layer = nn.TransformerEncoderLayer(
            self.config["d_model"], self.config["n_head"]
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.config["num_layers"]
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = self.transformer(outputs)  # mask 적용 필요함
        return outputs
