# AI-Based Dynamic Pricing & Revenue Optimization 🚀

## 🧠 Project Idea
This project builds an **AI system that determines the best delivery price** for each order. Instead of using a fixed price, the system:
* Predicts whether a customer will accept a price.
* Tests multiple price options.
* Selects the price that maximizes expected revenue.

---

## 🎯 Objective
The primary goal is to:

**Maximize Expected Revenue = Price × Probability of Acceptance**

---

## ⚙️ How the System Works

### 1. Data Processing
The system extracts key features:
* Distance
* Time (Peak / Off-peak)
* Base price

### 2. Market Simulation
We generate multiple price scenarios to simulate customer behavior.

### 3. Model Training
Two models are used:
* Logistic Regression (baseline)
* Deep Neural Network (main model)

---

## 📊 Results

### Model Performance

| Model | ROC-AUC | Brier Score |
|------|--------|------------|
| Logistic Regression | ~0.8185 | ~0.1728 |
| **Deep Neural Network** | **~0.8189** | **~0.1728** |

### Revenue Improvement

The system achieves approximately **~19.86% revenue improvement**.

---

## 📈 Visual Results

### ROC Curve
<img src="./Images/roc_curve.png" width="600"/>

### Confusion Matrix
<img src="./Images/confusion_matrix.png" width="600"/>

### Calibration Curve
<img src="./Images/calibration_curve.png" width="600"/>

### Revenue Impact
<img src="./Images/revenue.png" width="600"/>
---

## 📋 Sample Output (Top 10 Optimized Requests)

| distance_km | is_peak | p_base | Optimal_Price | Exp_Revenue | Strategy |
|-------------|---------|--------|---------------|-------------|----------|
| 3.484 | 0 | 10.788 | 8.853 | 5.916 | Demand Stimulation |
| 1.484 | 0 | 9.188 | 8.554 | 4.638 | Demand Stimulation |
| 2.276 | 0 | 9.820 | 8.330 | 5.102 | Demand Stimulation |
| 7.867 | 0 | 14.293 | 11.336 | 9.509 | Demand Stimulation |
| 5.573 | 0 | 12.459 | 9.881 | 7.586 | Demand Stimulation |
| 4.729 | 0 | 11.783 | 9.345 | 6.872 | Demand Stimulation |
| 6.080 | 0 | 12.864 | 10.202 | 8.005 | Demand Stimulation |
| 7.720 | 1 | 14.176 | 11.243 | 9.122 | Demand Stimulation |
| 3.076 | 0 | 10.460 | 8.873 | 5.636 | Demand Stimulation |
| 5.458 | 0 | 12.366 | 9.808 | 7.486 | Demand Stimulation |

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
