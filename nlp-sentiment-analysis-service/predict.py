import torch
from sentiment_model import SentimentModel
from transformers import BertTokenizer
from constants import MODEL_PATH, MAX_LEN
from data_loader import create_data_loader

class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentimentModel('HooshvareLab/bert-fa-base-uncased')
        self.model = self.model.load_from_checkpoint(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('HooshvareLab/bert-fa-base-uncased')
        
    def predict(self, comments):
        data_loader = create_data_loader(comments, None, self.tokenizer, MAX_LEN, BATCH_SIZE, None)
        predictions, _ = self._predict(data_loader)
        return predictions

    def _predict(self, data_loader):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                outputs = self.model(input_ids, attention_mask, token_type_ids)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds)

        return predictions

if __name__ == "__main__":
    predictor = SentimentPredictor()
    sample_comments = ["خیلی خوب بود", "اصلا خوشم نیامد"]
    print(predictor.predict(sample_comments))
