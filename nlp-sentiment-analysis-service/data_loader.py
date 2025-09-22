import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, comments, targets=None, tokenizer=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        inputs = {
            'comment': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        if self.targets is not None:
            inputs['targets'] = torch.tensor(self.targets[idx], dtype=torch.long)

        return inputs

def create_data_loader(comments, targets, tokenizer, max_len, batch_size):
    dataset = SentimentDataset(comments, targets, tokenizer, max_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
