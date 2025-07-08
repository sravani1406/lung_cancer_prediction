# 🫁 Lung Cancer Prediction Using Machine Learning

This project aims to predict the risk of lung cancer in patients based on demographic and health-related features using machine learning algorithms. It features many machine learning models, with a user-friendly interface built using **Streamlit**.

---

## 📌 Features

- ✅ Predict lung cancer risk based on patient inputs
- ✅ Dual-model support: Random Forest & XGBoost
- ✅ Handles data imbalance using **SMOTE**
- ✅ Clean UI built with Streamlit
- ✅ Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- ✅ Feature scaling and encoding for better model performance

---

## 🧠 Machine Learning Models Used

| Algorithm           | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| Logistic Regression | Simple linear classifier using a sigmoid function                     |
| KNN                 | Classifies based on nearest neighbors                                  |
| Decision Tree       | Splits data into branches for classification                          |
| SVM                 | Finds the optimal hyperplane to separate classes                      |
| Naive Bayes         | Probabilistic model based on Bayes' theorem                           |
| Random Forest       | Ensemble method using multiple decision trees                         |
| XGBoost             | Gradient boosting algorithm for high accuracy                         |

---

## 🧾 Dataset

- Source: Kaggle (Lung Cancer dataset)
- Features: Age, Gender, Smoking, Anxiety, Yellow fingers, etc.
- Target: Lung Cancer (YES/NO)

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Data Handling**: Pandas, NumPy
- **Model Persistence**: Joblib

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/lung-cancer-prediction.git
cd lung-cancer-prediction

2. Install Required Libraries

pip install -r requirements.txt

3. Train the Models (if not already trained)

python train_model.py

4. Launch the Streamlit App

streamlit run app.py


📊 Sample Results
Model	        Accuracy Precision	Recall	F1 Score
KNN	          92%	      91%	      90%	  90%
Random Forest	  88%	      87%	      85%	  86%
XGBoost	          90%	      89%	      88%	  88%


📜 License
This project is licensed under the MIT License.
