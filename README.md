
# 📰 Fake News Detection Using BERT

**Internship Project - Elementary Task**
**NOVANECTAR SERVICES PVT. LTD.**

## 📌 Project Overview

This project focuses on detecting fake news using a fine-tuned **BERT** (Bidirectional Encoder Representations from Transformers) model. The goal is to classify news articles as **Real** or **Fake** using Natural Language Processing techniques.

The dataset used is the **Fake and Real News Dataset** from Kaggle, which contains labeled real and fake news samples.

## ✅ Objectives

* Understand the problem of misinformation.
* Load and prepare real-world text data for binary classification.
* Fine-tune a pre-trained BERT model on this dataset.
* Evaluate model performance and test it with new sample headlines.

---

## 🧠 Technologies & Libraries Used

* Python
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* PyTorch
* Scikit-learn
* Pandas
* Google Colab
* KaggleHub for dataset access

---

## 📂 Dataset

**Name**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
**Source**: Kaggle
**Classes**:

* `1`: Real News
* `0`: Fake News

---

## 🚀 How to Run This Project

### 1. Clone or Upload Notebook to Google Colab

Open the notebook in Google Colab to ensure access to GPU and Kaggle integration.

### 2. Install Required Libraries

```bash
!pip install -q transformers datasets pandas scikit-learn kaggle torch
```

### 3. Upload `kaggle.json`

This enables downloading the dataset from Kaggle.

> You’ll be prompted to upload it when you run the notebook.

### 4. Download Dataset & Preprocess

* Downloads the CSV files from Kaggle.
* Combines and cleans data.
* Subsamples for fast training (optional).
* Prepares the dataset for BERT input.

### 5. Fine-Tune BERT

The model is trained for 2 epochs using `BertForSequenceClassification`.

### 6. Evaluate Performance

* Accuracy
* Classification Report
* Validation Loss

### 7. Predict New Headlines

You can test the model on your own inputs using the `predict_news()` function.

---

## 📊 Results

* **Accuracy**: \~95%+ (depending on dataset size and epochs)
* **Sample Test Predictions**:

  * "Scientists discover new planet..." → ✅ Real
  * "Government announces unicorns..." → ❌ Fake

---

## 💾 Saving and Exporting

The trained model and tokenizer are saved locally using:

```python
model.save_pretrained('./fake_news_bert_model/')
tokenizer.save_pretrained('./fake_news_bert_model/')
```

---

## 🔍 Example Predictions

```plaintext
"BREAKING: Government announces unicorns are real." → Fake News
"Stock market hits all-time high..." → Real News
```

---

## 🎓 Internship Note

This project is submitted as part of the **Elementary Task** for the **Artificial Intelligence Internship** at **NOVANECTAR SERVICES PVT. LTD.**. Successful completion of this task is required to receive the certificate.

---


