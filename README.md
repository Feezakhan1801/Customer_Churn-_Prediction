# Customer_Churn_Prediction

# 📊 Customer Churn AI Dashboard

An advanced **AI-powered Customer Churn Prediction Dashboard** built using Machine Learning, Explainable AI (SHAP), and a smart Recommendation Engine to help businesses reduce customer churn and improve retention strategies.

---

## 🚀 Live Features

* 🎯 Predict whether a customer will churn or not
* 📈 Churn Probability & Risk Level (Low / Medium / High)
* 📊 Retention Score Visualization
* 🤖 AI-Based Recommendations for customer retention
* 🧠 Explainable AI using SHAP (Feature Impact)
* 📄 Downloadable PDF Report
* 🎨 Modern Dashboard UI (Streamlit + Custom CSS)

---

## 🧠 Problem Statement

Customer churn is a critical problem in subscription-based businesses.
This project helps identify customers at risk of leaving and provides actionable insights to retain them.

---

## ⚙️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Models:** Scikit-learn
* **Explainability:** SHAP
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Deployment Ready:** Streamlit

---

## 🏗️ System Architecture

1. User inputs customer data via dashboard
2. Data is encoded using trained encoders
3. ML model predicts churn probability
4. Risk level & retention score calculated
5. AI recommendation engine suggests actions
6. SHAP explains feature importance
7. PDF report generated for download

---

## 🤖 Key Features Explained

### 🔮 Churn Prediction

Predicts whether a customer will leave based on behavioral and billing data.

### 📊 Retention Score

A custom score calculated as:

```
Retention Score = (1 - Churn Probability) * 100
```

### 🤖 Recommendation Engine

* High Risk → Discounts + Support
* Medium Risk → Engagement strategies
* Low Risk → Loyalty programs

### 🧠 Explainable AI (SHAP)

* Shows which features impact prediction
* Improves trust in ML model

### 📄 PDF Report Generator

* Generates professional churn report
* Includes prediction, probability, risk & recommendations

---

## 📸 Screenshots

👉 Add these screenshots:

* Dashboard UI
* Prediction result
* SHAP feature impact
* PDF report

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
│── app.py
│── models/
│   ├── model.pkl
│   ├── encoders.pkl
│── data/
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/Customer_Churn-_Prediction.git
cd Customer_Churn-_Prediction

pip install -r requirements.txt

streamlit run app.py
```

---

## 📈 Model Details

* Classification Model (e.g., Random Forest / Logistic Regression)
* Predicts binary outcome: Churn / No Churn
* Uses customer behavioral + billing features

---

## 💡 Key Insights

* High monthly charges increase churn probability
* Month-to-month contracts are high-risk
* Long tenure customers are more stable

---

## 🌐 Future Improvements

* 🌍 Deploy on Streamlit Cloud / AWS
* 📊 Add real-time data integration
* 🤖 Use advanced models (XGBoost, Deep Learning)
* 📱 Mobile-friendly dashboard

---

## 🙋‍♀️ Author

**Feeza Pathan**
Aspiring AI/ML Engineer 🚀

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
