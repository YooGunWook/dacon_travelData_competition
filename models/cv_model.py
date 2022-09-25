import torch
from torch import nn


class CVModel(nn.Module):
    def __init__(self, model, config):
        super(CVModel, self).__init__()
        self.model = model
        self.config = config
        if not self.config["cv_transformer"]:
            self.avgpool = nn.AvgPool2d(self.config["kernel_size"])
            self.dropout = nn.Dropout2d(self.config["dropout"])
        else:
            self.avgpool = nn.AvgPool1d(self.config["kernel_size"])
            self.dropout = nn.Dropout(self.config["dropout"])
        self.linear = nn.Linear(self.config["cv_dim"], 128)

    def forward(self, inputs):
        outputs = self.model.forward_features(inputs)
        if not self.config["cv_transformer"]:
            outputs = torch.flatten(self.dropout(self.avgpool(outputs)), start_dim=1)
        else:
            outputs = self.dropout(self.avgpool(outputs[:, 0, :]))
        outputs = self.linear(outputs)
        return outputs
