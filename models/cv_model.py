import torch
from torch import nn


class CVModel(nn.Module):
    def __init__(self, model, config):
        super(CVModel, self).__init__()
        self.model = model
        self.config = config
        self.avgpool = nn.AvgPool2d(self.config["kernel_size"])
        self.dropout = nn.Dropout2d(self.config["dropout"])
        self.linear = nn.Linear(self.config["cv_dim"], 128)

    def forward(self, inputs):
        outputs = self.model.forward_features(inputs)
        outputs = torch.flatten(self.dropout(self.avgpool(outputs)), start_dim=1)
        outputs = self.linear(outputs)
        return outputs
