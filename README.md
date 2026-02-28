# Dynamic Pricing Using Deep Learning  
### Revenue Optimization with Simulated Demand

---

## 📌 Overview

This project applies **Deep Learning** to a dynamic pricing problem in food delivery.

Using `historical_data.csv`, we simulate customer price behavior and train models to predict order acceptance and maximize revenue:

$$
Revenue = Price \times P(Accept)
$$

Since the dataset does not contain alternative price experiments, a structured economic simulator is built to generate training data.

---

## 🧠 Demand Simulation Model

### Distance Estimation

$$
Distance = \frac{Duration}{3600} \times 40
$$

### Base Price

$$
P_{base} = 8 + 0.8 \cdot Distance
$$

### Price Variation

$$
Price = P_{base} + \delta
$$

$$
\delta \sim \mathcal{U}(-0.5P_{base}, +0.5P_{base})
$$

---

### Nonlinear Price Sensitivity

Customer elasticity is modeled as:

$$
\alpha(X) = 0.15 + 0.05 \cdot Distance + 0.02 \cdot Distance^2 + 0.08 \cdot (Distance \cdot Rainy) - 0.07 \cdot Peak
$$
---

### Acceptance Probability

$$
P(Accept) =
\sigma\big(
-\alpha(X)(Price - P_{base}) + \epsilon
\big)
$$

$$
\epsilon \sim \mathcal{N}(0, 0.5)
$$

where $\sigma(\cdot)$ is the logistic function.

This produces a nonlinear demand surface suitable for supervised learning.

---

## 🤖 Models

- Logistic Regression (baseline)
- Deep Neural Network (nonlinear model)

The DNN captures complex elasticity patterns and improves predictive accuracy.

---

## 📈 Example Outputs

### ROC Curve

![ROC Curve](images/ROC.png)

The DNN achieves higher AUC compared to Logistic Regression.

---

### Example Predictions

![Random Sample](images/RANDOM.png)

The model predicts acceptance probability under different pricing scenarios.

---

## 🚀 How to Run

1. Install dependencies  
   ```bash
   pip install -r requirements.txt
