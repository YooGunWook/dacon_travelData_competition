import torch
from torch import nn


#########################          TOWER          #########################
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, kernel_list, num_filters, drop_rate):
        super(TextCNN, self).__init__()
        """Text Classification with CNN

        Args:
            embedding_dim (int): embedding dimension's size
            kernel_list (list): list of kernel size
            num_filters (int): filter size (CNN output size)
            num_classes (int): number of classes
            drop_rate (float): dropout rate
        """

        super(TextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.kernel_list = kernel_list
        self.num_filters = num_filters
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    stride=1,
                )
                for k in kernel_list
            ]
        )
        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(num_filters)
                for _ in kernel_list
            ]
        )
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(len(kernel_list) * num_filters, 128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = [F.relu(self.batch_norms[idx](conv(x))) for idx, conv in enumerate(self.conv)]
        x = [self.dropout(vec) for vec in x]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        feature = torch.cat(x, dim=1)
        x = self.fc(feature)
        return x, feature



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

#         self.linear = nn.Linear(self.config["d_model"], 128)
        self.textcnn = TextCNN(self.config["d_model"], [2,3,4,5], 256, 0.3)

    def forward(self, inputs):
        outputs = self.model(**inputs)["last_hidden_state"]
        attention_mask = inputs["attention_mask"].ne(1).bool()
        head_attention = []
        for i in range(attention_mask.shape[0]):
            tmp_attn = attention_mask[i].reshape(attention_mask.shape[1], 1)
            fin_attn = attention_mask[i] * tmp_attn
            head_attention.append(fin_attn.expand(self.config["n_head"], -1, -1))
        head_attention = torch.cat(head_attention)
#         outputs = self.transformer(outputs, mask=head_attention)[:, 0, :]
        outputs = self.transformer(outputs, mask=head_attention)
        outputs, _ = self.textcnn(outputs)
        return outputs
