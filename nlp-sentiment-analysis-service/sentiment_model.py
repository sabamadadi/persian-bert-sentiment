import torch.nn as nn
from transformers import BertModel

class SentimentModel(nn.Module):
    def __init__(self, model_name):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)
