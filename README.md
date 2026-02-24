# 🚚 Context-Aware Revenue Optimization  
### Dynamic Pricing for Food Delivery Using Deep Learning

**Abdulrahman Jaber Ageeli**  
February 2026  

---

## 📌 Overview

This project builds a **context-aware dynamic pricing system** for food delivery platforms.

Instead of using static delivery fees, we simulate customer behavior under different pricing scenarios and train machine learning models to:

- Predict order acceptance probability  
- Model nonlinear price sensitivity  
- Identify revenue-maximizing delivery prices  

The result is a fully reproducible pricing optimization pipeline combining economic structure with deep learning.

---

## 💡 Business Problem

Delivery platforms face a trade-off:

- Higher prices → higher margin  
- Higher prices → lower acceptance rate  

The goal is to find the optimal balance:

\[
\text{Revenue} = \text{price} \times P(\text{accept})
\]

We estimate \( P(\text{accept}) \) using contextual features and machine learning.

---

## 🧩 Key Features

### Context-Aware Demand Modeling
Customer price sensitivity varies with:

- Distance  
- Peak hours  
- Weather conditions  

We simulate nonlinear elasticity using:

\[
\alpha(X) =
0.15
+ 0.05 \cdot \text{distance}
+ 0.02 \cdot \text{distance}^2
+ 0.08 \cdot (\text{distance} \times \text{is\_rainy})
- 0.07 \cdot \text{is\_peak}
\]

---

### Stochastic Demand Simulation

For each order, multiple counterfactual prices are generated:

\[
\text{price} = P_{\text{base}} + \delta
\]

\[
\delta \sim \mathcal{U}(-0.5P_{\text{base}}, +0.5P_{\text{base}})
\]

Acceptance probability:

\[
P(\text{accept}) = \sigma\left(-\alpha(X)(\text{price} - P_{\text{base}}) + \epsilon \right)
\]

This produces a realistic nonlinear demand surface.

---

## 🤖 Models

### 1️⃣ Logistic Regression (Baseline)
- Linear decision boundary  
- Fast and interpretable  

### 2️⃣ Deep Neural Network
- Architecture: 64 → 32 → 16  
- ReLU activations  
- Dropout (0.2)  
- Early stopping  

The DNN captures nonlinear interactions embedded in contextual elasticity.

---

## 📊 Results

| Model | AUC | Brier Score |
|-------|------|------------|
| Logistic Regression | ~0.945 | ~0.091 |
| Deep Neural Network | ~0.957 | ~0.084 |

### Key Takeaways

- DNN improves predictive performance
- Nonlinear modeling increases revenue optimization accuracy
- Revenue curve identifies a clear optimal price
- No major overfitting observed

---

## 🛠 Engineering Highlights

- Proper train/test split before price expansion (no leakage)
- Reproducible stochastic simulation
- Revenue-driven evaluation (not just classification metrics)
- Clean modular pipeline

---

## 📁 Dataset

Source: Kaggle DoorDash dataset  
File required:

Used features:
- Order timestamp  
- Driving duration  

Derived features:
- Distance  
- Peak indicator  
- Simulated weather  

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
DL_Project.ipynb
