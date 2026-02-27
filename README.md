# 🕵️ Predicting Fraudulent Job Postings

A data science project that uses machine learning to classify job postings as either legitimate or fraudulent. Built using Kaggle's Real/Fake Job Posting Prediction dataset, the project covers the full data science pipeline — from data cleaning and EDA to training and evaluating five different classification models.

📰 **Read the full write-up on Medium:** [Predicting Fraudulent Job Postings](https://faiziali.medium.com/predicting-fraudulent-job-postings-af0056251533)

---

## 📌 Problem Statement

Fraudulent job postings are a growing problem in the digital age, leading to scams, identity theft, and financial loss for unsuspecting job seekers. This project aims to build robust classifiers capable of automatically detecting fake job postings based on their textual and categorical features.

---

## 📊 Dataset

**Source:** [Kaggle – Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

- ~18,000 job postings
- 866 flagged as fraudulent (~4.8%)
- 17 features including textual (title, description, requirements), categorical (employment type, industry, education), and binary (has_company_logo, has_questions) fields
- Target variable: `fraudulent` (0 = Legitimate, 1 = Fake)

---

## 🔬 Research Questions

1. Are fraudulent job postings more common in certain locations or departments?
2. Are there specific keywords or phrases common in fraudulent postings?
3. Which industries are most affected by fake job postings?
4. Are fraudulent jobs less likely to have a company logo?
5. Are certain employment types, experience levels, or education levels more associated with fraud?

---

## 🛠️ Project Pipeline

### 1. Data Cleaning & Preprocessing
- Dropped `salary_range` and `department` columns due to excessive missing values
- Converted all text to lowercase
- Applied a custom `clean_text()` function that:
  - Removes numbers, punctuation, HTML tags, and URLs
  - Normalizes contractions (e.g. "I've" → "I have")
  - Tokenizes text, removes stopwords, and applies Porter Stemming
- Removed outliers based on character count using IQR
- Balanced the dataset by downsampling legitimate postings to match the count of fraudulent ones (~866 each)

### 2. Feature Engineering
- Combined text columns (`title`, `company_profile`, `description`, `requirements`, `benefits`) into a single `text` feature
- Applied **TF-IDF Vectorization** (top 100 features) to convert text into numeric form
- Applied **Label Encoding** to categorical features for the KNN model

### 3. Exploratory Data Analysis (EDA)
Key findings:
- Australia has a disproportionately high rate of fraudulent postings
- Fraudulent postings are most common in the **Oil & Energy** industry
- Fake postings tend to target **entry-level**, **full-time** roles
- Most fake postings require only a **high school diploma or equivalent**
- Job postings **without a company logo** are significantly more likely to be fraudulent
- Fraudulent postings tend to have a **shorter character count** than legitimate ones
- Top keywords in fake postings: *manage, service, experience, customer, skill, develop, job, and, position*

---

## 🤖 Models Trained

All text-based models were trained on a balanced dataset with a 70/30 train-test split.

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 78% | 78% | 78% |
| Random Forest (Entropy) | 89% | 90% | 90% |
| **Random Forest (Gini)** | **90%** | **90%** | **87%** |
| Support Vector Machine | 79% | 78% | 83% |
| Multinomial Naive Bayes | 67% | 69% | — |
| K-Nearest Neighbors* | 81% | 78% | — |

> *KNN was trained exclusively on non-text categorical features (employment type, experience, education, industry, function).

**Best Model: Random Forest with Gini Impurity** — highest accuracy with the strongest overall balance of precision, recall, and F1-score.

---

## 📁 Project Structure

```
FraudulentJobPostings/
│
├── DS_FinalProject_Group8.ipynb   # Main notebook (EDA + all models)
├── fake_job_postings.csv          # Dataset (download from Kaggle)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly nltk joblib
```

### NLTK Downloads (run once)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Running the Notebook

The notebook was originally developed in **Google Colab** with the dataset stored in Google Drive. To run locally:

1. Download `fake_job_postings.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
2. Place it in the same directory as the notebook
3. Replace the Google Drive mount cell with:
```python
df = pd.read_csv("fake_job_postings.csv")
```
4. Run all cells sequentially

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn / Plotly | Visualization |
| NLTK | Text preprocessing & stemming |
| Scikit-learn | TF-IDF, model training & evaluation |
| Google Colab | Development environment |

---

## 📝 Key Takeaways

- Class imbalance is a major challenge — the raw dataset is ~95% legitimate, requiring deliberate balancing before training
- Text features are highly informative for fraud detection; TF-IDF combined with Random Forest performs best
- Non-text categorical features alone yield ~81% accuracy with KNN, showing that even structural signals carry meaningful predictive power
- Fraudulent postings tend to be shorter, logo-free, entry-level, and concentrated in specific industries

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
