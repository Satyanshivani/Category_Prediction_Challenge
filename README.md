# Comment Category Prediction Challenge

## 📌 Overview
This project focuses on predicting the final category of online comments using **Machine Learning** and **Natural Language Processing (NLP)** techniques.

The dataset contains:
- Comment text
- User engagement information
- Emoticon-based features
- Hidden platform indicators
- Category-related metadata

The main objective is to classify comments into the correct label category by analyzing both textual and structured data.

---

# 📂 Dataset Information

The competition dataset contains three files:

| File Name | Description |
|-----------|-------------|
| `train.csv` | Training dataset with target labels |
| `test.csv` | Test dataset without labels |
| `Sample.csv` | Sample submission format |

---

# 🧾 Features Description

| Feature | Description |
|---------|-------------|
| `comment` | Raw text content of the comment |
| `created_date` | Date and time of comment creation |
| `post_id` | Unique post identifier |
| `emoticon_1` | Emoticon feature group 1 |
| `emoticon_2` | Emoticon feature group 2 |
| `emoticon_3` | Emoticon feature group 3 |
| `upvote` | Number of positive reactions |
| `downvote` | Number of negative reactions |
| `if_1` | Internal hidden feature 1 |
| `if_2` | Internal hidden feature 2 |
| `race` | Race-related indicator |
| `religion` | Religion-related indicator |
| `gender` | Gender-related indicator |
| `disability` | Disability-related indicator |
| `label` | Final target category |

---

# 🛠️ Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
scipy
```

---

# 🔍 Project Workflow

## 1️⃣ Data Loading & Overview
- Loaded training and testing datasets
- Checked dataset shape and datatypes
- Analyzed missing values
- Generated statistical summaries

### Dataset Summary
- Total Rows: **198,000**
- Features: **15**
- Target Classes: **4**

---

# 📊 Exploratory Data Analysis (EDA)

## ✅ Statistical Analysis
Performed:
- Average upvote/downvote analysis
- Comment length analysis
- Label-wise comparison

### Key Insights
- Label 1 comments had the highest average comment length
- Label 3 comments were comparatively shorter
- Upvote and downvote values varied across labels

---

## 📈 Visualizations

### 🎯 Target Distribution
Created countplots to analyze class balance.

### Insights
- Dataset is imbalanced
- Label 0 contains the highest number of samples
- Labels 1 and 3 contain fewer samples

Because of imbalance, **Weighted F1-score** was used for evaluation.

---

### 🔗 Correlation Heatmap
Analyzed relationships between numeric features.

### Insights
- Most features showed weak correlation
- Moderate correlation observed between:
  - `upvote`
  - `downvote`

---

### 📦 Outlier Detection
Used boxplots for detecting extreme values.

### Insights
- Upvote feature contained many outliers
- Extreme values were capped later using IQR method

---

# 🧹 Data Cleaning & Preprocessing

The following preprocessing steps were applied:

✅ Removed duplicate rows

✅ Filled missing categorical values using:
```python
'None'
```

✅ Filled missing numeric values using median

✅ Removed links and special characters from comments

✅ Converted text to lowercase

✅ Removed extra spaces

✅ Capped outliers using IQR method

---

# ⚙️ Feature Engineering

Multiple feature engineering techniques were used to improve model performance.

## 🔤 Text Features
### TF-IDF Word Level
- Max Features: 30,000
- N-grams: (1,2)

### TF-IDF Character Level
- Max Features: 15,000
- Character N-grams: (2,4)

---

## 🏷️ Categorical Features
Used:
```python
OneHotEncoder
```

for encoding categorical columns.

---

## 📏 Numeric Features
Used:
```python
MaxAbsScaler
```

for scaling sparse numeric features.

---

## 🔗 Final Feature Matrix
Combined:
- Word TF-IDF features
- Character TF-IDF features
- Numeric features
- Encoded categorical features

using:
```python
hstack()
```

---

# 🤖 Machine Learning Models

Multiple models were trained and evaluated.

| Model | Validation F1 Score |
|-------|--------------------|
| SGD Classifier | 0.767 |
| Linear SVM | 0.789 |
| Logistic Regression | 0.808 |
| Random Forest | 0.852 |
| LightGBM | 0.917 |

---

# 🏆 Best Performing Model

## 🚀 LightGBM
LightGBM achieved the best individual model performance.

### Why it performed well?
- Handles sparse data efficiently
- Works well with large datasets
- Captures complex feature interactions
- Fast training speed

### Best Validation F1 Score
```python
0.9175
```

---

# 🔥 Ensemble Learning

To further improve performance, ensemble learning was applied.

## Models Combined
- Logistic Regression
- Random Forest
- LightGBM

## Weighted Averaging
```python
0.2 * LR
0.2 * RF
0.6 * LGBM
```

### Final Ensemble Score
```python
0.9196
```

The ensemble model outperformed all individual models.

---

# 📉 Overfitting Analysis

Train and validation F1-scores were compared.

### Observations
- Random Forest showed high overfitting
- SGD had the smallest train-validation gap
- LightGBM provided the best balance between performance and generalization

---

# 🏁 Final Submission

Final steps:
- Retrained models on full dataset
- Generated predictions on test data
- Created final submission file

Generated File:
```python
submission.csv
```

---

# ▶️ How to Run the Project

## Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
```

## Run Notebook

```bash
jupyter notebook code.ipynb
```

---

# 📁 Project Structure

```bash
Category_Prediction_Challenge/
│
├── train.csv
├── test.csv
├── Sample.csv
├── code.ipynb
├── submission.csv
└── README.md
```

---

# 📌 Key Learnings

- Text preprocessing for NLP
- Feature engineering using TF-IDF
- Sparse matrix handling
- Model comparison techniques
- Ensemble learning
- Handling imbalanced datasets
- LightGBM optimization

---

# 👩‍💻 Author

## Satyanshi Singh

GitHub Repository:

:contentReference[oaicite:0]{index=0}

---
