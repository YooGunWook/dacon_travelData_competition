import torch
from torch import nn


class MultiModalClassifier(nn.Module):
    def __init__(self, nlp_model, cv_model, config):
        super(MultiModalClassifier, self).__init__()
        self.nlp_model = nlp_model
        self.cv_model = cv_model
        self.layer = nn.Linear(256, config["num_classes"])
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, nlp_inputs, cv_inputs):
        nlp_output = self.nlp_model(nlp_inputs)
        cv_output = self.cv_model(cv_inputs)
        concat_output = torch.cat([nlp_output, cv_output], dim=1)
        concat_output = self.dropout(self.relu(concat_output))
        output = self.layer(concat_output)
        return output, concat_output
