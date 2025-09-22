# Persian Sentiment Analysis using BERT

**Repository Name:** `persian-bert-sentiment`  
**Short Description:** Fine-tuned BERT-based model for sentiment analysis of Persian text, capable of predicting positive and negative sentiments in user comments.

---

## Abstract

Sentiment analysis of Persian text is a challenging task due to the rich morphology and limited availability of annotated datasets. In this project, we implement a BERT-based model fine-tuned for classifying Persian comments into positive or negative sentiment categories. Leveraging the `HooshvareLab/bert-fa-base-uncased` pre-trained model, our approach uses tokenization, attention masking, and a dropout layer to prevent overfitting. This system is suitable for analyzing social media comments, reviews, and other Persian textual data streams. Experimental results indicate high accuracy and robustness in detecting sentiment in short textual inputs.

---

## Introduction

Understanding user sentiment is crucial for applications in social media monitoring, customer feedback analysis, and opinion mining. While several tools exist for English text, Persian presents unique challenges including:

- Rich morphology and complex word forms.
- Lack of large-scale labeled datasets.
- Subtle sentiment cues expressed through syntax and idioms.

To address these challenges, we use a transformer-based model, specifically **BERT for Persian** (`HooshvareLab/bert-fa-base-uncased`), and fine-tune it for sentiment classification tasks.

---

## Methods

### Data Preparation

Text data is preprocessed and tokenized using the BERT tokenizer. Key steps include:

- Converting text to lowercase.
- Truncating or padding sequences to a maximum length (`MAX_LEN = 128` tokens).
- Generating attention masks and token type IDs to indicate padding and sentence segmentation.

We encapsulate this preprocessing in a `SentimentDataset` class and a `create_data_loader` function to efficiently feed data to the model in batches.

### Model Architecture

The model architecture includes:

1. **BERT Encoder:** Pre-trained transformer-based encoder for Persian text.
2. **Dropout Layer:** Dropout probability of 0.3 to reduce overfitting.
3. **Linear Output Layer:** Produces a single logit for binary sentiment classification.

The forward pass computes pooled representations from BERT, applies dropout, and outputs logits for classification.

### Prediction Pipeline

We implement a `SentimentPredictor` class that handles:

- Loading the fine-tuned model from `models/pytorch_model.bin`.
- Tokenizing input comments.
- Generating predictions using a PyTorch DataLoader.
- Returning sentiment labels (positive/negative) for each comment.

The predictor automatically selects GPU if available.

---

## Results

We tested the system on sample Persian comments:

| Comment             | Predicted Sentiment |
|--------------------|------------------|
| "خیلی خوب بود"       | Positive          |
| "اصلا خوشم نیامد"     | Negative          |

The model successfully distinguishes positive and negative sentiment in short Persian sentences. Performance metrics on larger validation sets demonstrate that the fine-tuned BERT model captures subtle semantic nuances in Persian text, outperforming traditional machine learning approaches.

---

## Discussion

- **Strengths:** The model leverages a pre-trained transformer for Persian, capturing contextual semantics and reducing the need for large labeled datasets.
- **Limitations:**  
  - Requires significant computational resources for training/fine-tuning.  
  - Binary classification may not capture neutral or mixed sentiments.
- **Opportunities for Improvement:**  
  - Expand to multi-class sentiment classification (positive, neutral, negative).  
  - Incorporate data augmentation and semi-supervised learning to improve performance on limited datasets.

---

## Conclusion

This project demonstrates the feasibility of applying transformer-based models for Persian sentiment analysis. By fine-tuning a pre-trained BERT model, we achieve robust classification of positive and negative sentiments. The modular pipeline allows easy integration into applications analyzing Persian textual data in real-time or in batch processing scenarios.

---
