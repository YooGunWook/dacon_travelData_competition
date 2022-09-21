import torch
from torch import nn


class NLPModel(nn.Module):
    def __init__(self, model, config):
        super(NLPModel, self).__init__()
        self.model = model
        self.config = config
        self.encoder_layer = nn.TransformerEncoderLayer(
            self.config["d_model"], self.config["n_head"], batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.config["num_layers"]
        )
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(self.config["d_model"])
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.config["d_model"], 128)

    def forward(self, inputs):
        outputs = self.model(**inputs)["last_hidden_state"]
        attention_mask = inputs["attention_mask"].ne(1).bool()
        head_attention = []
        for i in range(attention_mask.shape[0]):
            tmp_attn = attention_mask[i].reshape(attention_mask.shape[1], 1)
            fin_attn = attention_mask[i] * tmp_attn
            head_attention.append(fin_attn.expand(self.config["n_head"], -1, -1))
        head_attention = torch.cat(head_attention)
        outputs = self.transformer(outputs, mask=head_attention)[:, 0, :]  # mask 적용 필요함
        outputs = self.dropout(self.relu(self.batch(outputs)))
        outputs = self.linear(outputs)
        return outputs
