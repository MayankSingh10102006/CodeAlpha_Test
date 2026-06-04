# 🏥 AI-Powered Diabetes Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Classifier-green?style=for-the-badge)

**An intelligent machine learning web application that predicts diabetes risk using patient medical data.**

*CodeAlpha Machine Learning Internship — Task 4: Disease Prediction*

</div>

---

## 📌 Project Overview

This project is a production-ready **AI-Powered Diabetes Prediction Dashboard** built using Python, Streamlit, and Scikit-learn. It uses the **Pima Indians Diabetes Dataset** to train a Random Forest Classifier that predicts whether a patient is at risk of diabetes based on 8 medical parameters.

The application features a modern dark-themed hospital-style UI with interactive charts, real-time predictions, medical recommendations, and prediction history tracking.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔬 **Real-time Prediction** | Instant diabetes risk prediction with probability score |
| 📊 **Analytics Dashboard** | Outcome distribution, confusion matrix, ROC curve |
| 🧠 **Model Insights** | Feature importance & correlation heatmap |
| 📋 **Prediction History** | Track and export all predictions as CSV |
| 📥 **PDF Report** | Download detailed prediction report |
| 🎨 **Modern UI** | Dark hospital-themed responsive design |
| ⚙️ **Live Indicators** | Real-time glucose & BMI gauges |

---

## 🖥️ Screenshots

### Prediction Dashboard
> Patient input sliders with normal medical ranges and real-time risk gauge

### Analytics Tab
> Outcome distribution, confusion matrix, ROC curve, and feature distribution charts

### Model Insights
> Feature importance bar chart and feature correlation heatmap

### History Tab
> Session-based prediction tracking with CSV export

---

## 🤖 Machine Learning Model

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Classifier |
| **n_estimators** | 200 |
| **max_depth** | 10 |
| **Test Size** | 20% |
| **Preprocessing** | StandardScaler + Median Imputation |
| **Accuracy** | ~75.3% |
| **ROC-AUC Score** | ~0.815 |

### Evaluation Metrics
- ✅ Accuracy Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix
- ✅ Classification Report (Precision, Recall, F1)

---

## 📁 Dataset

**Name:** Pima Indians Diabetes Database  
**Source:** UCI Machine Learning Repository / Kaggle  
**Records:** 768 patients  
**Features:** 8 medical attributes  
**Target:** Diabetic (1) / Non-Diabetic (0)

### Features Used

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| Pregnancies | Number of pregnancies | 0–5 |
| Glucose | Plasma glucose concentration (mg/dL) | 70–140 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 60–90 |
| SkinThickness | Triceps skin fold thickness (mm) | 10–40 |
| Insulin | 2-hour serum insulin (mu U/ml) | 16–166 |
| BMI | Body Mass Index | 18.5–24.9 |
| DiabetesPedigreeFunction | Genetic likelihood of diabetes | Lower is better |
| Age | Patient age (years) | — |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/MayankSingh10102006/CodeAlpha_Test.git
cd CodeAlpha_Test
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn plotly fpdf2
```

**4. Add the dataset**

Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the project root.

**5. Run the application**
```bash
streamlit run app.py
```

**6. Open in browser**
```
http://localhost:8501
```

---

## 📦 Project Structure

```
CodeAlpha_Test/
│
├── app.py                  # Main Streamlit application
├── diabetes.csv            # Dataset (Pima Indians Diabetes)
├── model.pkl               # Trained Random Forest model
├── scaler.pkl              # Fitted StandardScaler
└── README.md               # Project documentation
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly, Custom CSS
- **Backend:** Python 3.9
- **ML Libraries:** Scikit-learn, NumPy, Pandas
- **Visualization:** Plotly Express, Plotly Graph Objects
- **PDF Generation:** fpdf2
- **Model Persistence:** Pickle

---

## 📊 How It Works

```
1. Load Dataset (diabetes.csv)
        ↓
2. Preprocess Data
   • Replace 0s with NaN for invalid columns
   • Fill NaN with median values
   • StandardScaler normalization
        ↓
3. Train Random Forest Classifier
   • 200 estimators, max_depth=10
   • 80/20 train-test split
        ↓
4. Save Model (model.pkl + scaler.pkl)
        ↓
5. User Inputs Parameters via Streamlit
        ↓
6. Apply same preprocessing → Predict
        ↓
7. Display Result + Recommendations
```

---

## ⚠️ Disclaimer

> This application is built for **educational purposes only** as part of the CodeAlpha ML Internship program. It is **not a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare provider for medical decisions.

---

## 👨‍💻 Developer

**Mayank Singh**  
🎓 Machine Learning Intern @ CodeAlpha  
🔗 [GitHub](https://github.com/MayankSingh10102006)

---

## 🏢 About CodeAlpha

CodeAlpha is a leading software development company driving innovation through AI and intelligent systems. This project was developed as **Task 4** of the CodeAlpha Machine Learning Internship program.

🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ and Python

</div>
