# Zero-Shot vs Fine-Tuned Approaches for Phishing Email Detection

## 📌 Project Overview
This project compares **Zero-Shot Learning** and **Fine-Tuned Transformer-based models** for detecting phishing emails using Natural Language Processing (NLP). The goal is to evaluate whether a general-purpose pre-trained model can reliably detect phishing emails without task-specific training, and how its performance compares to a fine-tuned model trained on labeled phishing data.

This project was completed as part of an **NLP academic assignment** and focuses on cybersecurity-related text classification.

---

## 🎯 Objective
- To evaluate the effectiveness of **Zero-Shot classification** for phishing email detection.
- To fine-tune a transformer model on a phishing dataset and analyze performance improvements.
- To experimentally prove whether fine-tuning is necessary for high-stakes NLP tasks like phishing detection.

---

## 📊 Dataset
- **Source:** zefang-liu/phishing-email-dataset (Hugging Face)
- **Total Emails Used:** 14,640
- **Classes:**  
  - Phishing  
  - Legitimate
- **Preprocessing Steps:**
  - Removed noise and irrelevant entries
  - Balanced the dataset using undersampling
  - 80% training set, 20% test set

> ⚠️ Dataset files are excluded from this repository due to large size.  
> Please download them directly from Hugging Face if needed.

---

## 🧠 Models Used

### 1️⃣ Zero-Shot Model
- **Model:** facebook/bart-large-mnli
- **Approach:**  
  Used Natural Language Inference (NLI) without any phishing-specific training.

### 2️⃣ Fine-Tuned Model
- **Model:** distilbert-base-uncased
- **Approach:**  
  Fine-tuned on labeled phishing emails using Hugging Face Trainer API.

---

## ⚙️ Training Configuration (Fine-Tuning)
- Epochs: 2  
- Batch Size: 16  
- Framework: PyTorch + Hugging Face Transformers

---

## 📈 Results

| Approach | Accuracy | Precision (Phishing) | Recall (Phishing) | F1-Score |
|--------|----------|----------------------|-------------------|----------|
| Zero-Shot | 54.81% | 61.0% | 26.0% | — |
| Fine-Tuned | **97.61%** | **96.52%** | **98.77%** | **97.63%** |

---

## 🔍 Key Observations
- Zero-Shot classification failed to detect most phishing emails due to lack of domain-specific knowledge.
- Fine-tuning enabled the model to learn malicious intent patterns such as urgency, suspicious requests, and context mismatch.
- Fine-Tuned DistilBERT significantly outperformed Zero-Shot BART across all evaluation metrics.

---

## 🏁 Conclusion
This project demonstrates that **Zero-Shot learning is insufficient for cybersecurity tasks** such as phishing detection. While Zero-Shot models understand general language, they fail to capture deceptive patterns without task-specific training.

**Final Verdict:**  
Fine-Tuning is not optional—it is **mandatory** for achieving reliable, production-grade results in phishing email detection.

---

## 🛠 Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- NLP, Text Classification

---

## 📄 Project Files
- `nlp-as-3-finetunning.ipynb` → Fine-tuning and evaluation notebook
- `Project Report` → Detailed comparison and analysis (included in assignment document)

---

## 👨‍🎓 Author
**Muhammad Wasif**  
BS Artificial Intelligence  
NLP Assignment — Semester 7

---

## 🔗 Repository Link
https://github.com/rai-wasif/Zeroshot-vs-Finetunning
