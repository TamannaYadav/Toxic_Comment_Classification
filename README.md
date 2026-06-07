# 🧠 Toxic Comment Detection using ML & Deep Learning

## 📌 Project Overview
This project focuses on detecting toxic comments in text data using Machine Learning and Deep Learning models. The system classifies comments as toxic or non-toxic and helps improve online safety through automated content moderation.

We experimented with multiple approaches:
- TF-IDF + Logistic Regression (Baseline Model)
- BiLSTM (Deep Learning Model)
- BiGRU (Optimized Sequential Model)
- BERT (Transformer-based model – in progress)

---

## Dataset

Jigsaw Toxic Comment Classification Challenge

Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

---

## 📂 Repository Structure
```bash
01_EDA_Visualization.ipynb
02_preprocessing_tfidf.ipynb
03_Tfidf_model.ipynb
LSTM_GRU.ipynb
text_analysis.ipynb
preprocessing_bert.py
preprocessing_tfidf.py
text_preprocessing_gru.py
requirements.txt
README.md
```
---

## ⚙️ Workflow

### 1. Data Preprocessing
- Remove URLs, emails, special characters
- Convert text to lowercase
- Tokenization
- Padding sequences for deep learning models
  
### 2. Feature Enineering
- TF-IDF vectorization for ML models
- Word embeddings for deep learning models
- BERT tokenizer for transformer model

### Models used
--> TF-IDF + Logistic Regression (Baseline)
- Fast and simple
- Limited contextual understanding

--> BiLSTM Model
- Captures sequential dependencies
- Better than traditional ML models

--> BiGRU (Best Model)
- Faster than LSTM
- Best balance of speed and accuracy

--> BERT Model
- Uses pre-trained contextual embeddings
- Minimal preprocessing required
- Currently under training

---

## Model Performance

### TF-IDF + Logistic Regression

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 92%   |
| Precision | 0.78  |
| Recall    | 0.72  |
| F1 Score  | 0.75  |
| ROC-AUC   | 0.94  |

### BiGRU

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 96%    |
| Precision | 0.8313 |
| Recall    | 0.7834 |
| F1 Score  | 0.8066 |
| ROC-AUC   | 0.9765 |

---

## 📊 Key Insights

- BiGRU performed best among all models
- Dataset is highly imbalanced (more non-toxic comments)
- Deep learning improves recall for toxic class
- TF-IDF is a strong baseline but lacks semantic understanding

---

## 🚀 How to Run

1. Clone repo
```bash
git clone https://github.com/your-username/toxic-comment-detection.git
cd toxic-comment-detection
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run notebooks
```bash
jupyter notebook
```

---

## 📌 Conclusion

This project shows that deep learning models, especially BiGRU, outperform traditional ML methods in toxic comment detection. BERT is expected to further improve results once fully trained.
