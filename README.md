# 🚚 Context-Aware Revenue Optimization  
### Dynamic Pricing for Food Delivery Using Deep Learning  

**Abdulrahman Jaber Ageeli**  
February 2026  

---

## 📌 Overview

This project builds a context-aware dynamic pricing system for food delivery platforms.

Instead of using static delivery fees, we simulate customer behavior under different pricing scenarios and train machine learning models to:

- Predict order acceptance probability  
- Model nonlinear price sensitivity  
- Identify revenue-maximizing delivery prices  

The result is a fully reproducible pricing optimization pipeline combining economic structure with deep learning.

---

## 💡 Business Objective

Delivery platforms face a pricing trade-off:

- Higher prices → higher margin  
- Higher prices → lower acceptance rate  

The goal is to maximize expected revenue:

$$
Revenue = price \times P(accept)
$$

We estimate $P(accept)$ using contextual features and machine learning.

---

## 📊 Dataset

Source: Kaggle DoorDash dataset  

Required file:

Selected variables:

- `created_at`
- `estimated_store_to_consumer_driving_duration`

Derived contextual features:

- Distance  
- Peak hour indicator  
- Simulated weather  

---

## ⚙️ Methodology

### 1️⃣ Distance Estimation

Driving duration is converted into distance (km):

$$
distance = \frac{duration\_seconds}{3600} \times 40
$$

Assuming 40 km/h average urban speed.

---

### 2️⃣ Peak Indicator

Peak hours:

- Lunch: 11:00–14:00  
- Dinner: 18:00–21:00  

Binary variable: `is_peak`.

---

### 3️⃣ Weather Simulation

Rain probability is simulated conditionally:

$$
P(rain) = 0.2 + 0.1 \times is\_peak
$$

This increases contextual variability during high-demand periods.

---

### 4️⃣ Base Pricing Model

Instead of a fixed fee:

$$
P_{base} = 8 + 0.8 \times distance
$$

This captures:

- Fixed operational cost (8)  
- Distance-based marginal cost  

---

### 5️⃣ Stochastic Demand Simulation

For each order, multiple counterfactual prices are generated:

$$
price = P_{base} + \delta
$$

$$
\delta \sim U(-0.5 P_{base}, +0.5 P_{base})
$$

Minimum price is constrained to \$2.

---

### 6️⃣ Nonlinear Price Sensitivity

Elasticity varies by context:

$$
\alpha(X) =
0.15
+ 0.05 \cdot distance
+ 0.02 \cdot distance^2
+ 0.08 \cdot (distance \times is\_rainy)
- 0.07 \cdot is\_peak
$$

This introduces:

- Quadratic distance effects  
- Distance–weather interaction  
- Reduced elasticity during peak periods  

---

### 7️⃣ Acceptance Probability

First compute:

$$
z = -\alpha(X)(price - P_{base}) + \epsilon
$$

Then:

$$
P(accept) = \sigma(z)
$$

where:

$$
\epsilon \sim N(0, 0.5)
$$

This produces a realistic nonlinear demand surface.

---

## 🤖 Models

### Logistic Regression
- Linear decision boundary  
- Baseline comparison  

### Deep Neural Network
Architecture:
- 64 → 32 → 16 hidden units  
- ReLU activations  
- Dropout (0.2)  
- Early stopping  

The DNN captures nonlinear contextual interactions embedded in $\alpha(X)$.

---

## 📈 Results

| Model | AUC | Brier Score |
|-------|------|------------|
| Logistic Regression | ~0.945 | ~0.091 |
| Deep Neural Network | ~0.957 | ~0.084 |

### Key Findings

- Deep learning improves discrimination and calibration.
- Nonlinear modeling enhances revenue optimization.
- A clear revenue-maximizing price emerges from the model.
- No significant overfitting observed.

---

## 🛠 Engineering Strengths

- Train/test split performed before price expansion (prevents leakage)
- Reproducible stochastic simulation
- Revenue-driven evaluation
- Clean modular pipeline

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
