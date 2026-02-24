# DL_Project  
## Context-Aware Revenue Optimization in Food Delivery Platforms  
### Using Deep Learning and Stochastic Demand Simulation  

**By Student – June 2026**

---

## 📌 Project Overview
Food delivery platforms operate in highly dynamic environments where demand depends on context such as distance, time, and weather.

This project proposes a **context-aware revenue optimization framework** that combines:

- Stochastic demand simulation  
- Machine learning models  
- Deep neural networks for nonlinear pricing behavior  

The goal is to estimate order acceptance probability and find the revenue-maximizing delivery price.

---

## 📊 Dataset
We use the Kaggle DoorDash dataset:

`historical_data.csv`

Key variables:
- Order timestamp  
- Store-to-consumer duration  
- Derived contextual features  

---

## ⚙️ Methodology

### Feature Engineering
- Distance estimation from driving duration  
- Peak hour indicator  
- Simulated weather conditions  

### Pricing Model
Base price:


### Demand Simulation
- Counterfactual price generation  
- Nonlinear contextual sensitivity  
- Logistic acceptance model  

### Models Used
- Logistic Regression (baseline)  
- Deep Neural Network (64 → 32 → 16, Dropout, Early Stopping)

---

## 📈 Evaluation Metrics
- ROC-AUC  
- Brier Score  
- Revenue optimization curve  

---

## 🧠 Results
The Deep Neural Network outperformed Logistic Regression in both:

- Predictive accuracy  
- Revenue optimization performance  

---

## 🚀 How to Run

1. Install dependencies


2. Open notebook


3. Run all cells

---

## 📚 References
- Bishop (2006) – Pattern Recognition and Machine Learning  
- Goodfellow et al. (2016) – Deep Learning  
- Friedman et al. (2001) – Elements of Statistical Learning  
- McFadden (1974) – Conditional Logit Model  

---

## 👨‍💻 Author
Student – Abdulrahman Jaber Ageeli Introduction to Deep Learning Project
