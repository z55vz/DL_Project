# DL_Project  
## Context-Aware Revenue Optimization in Food Delivery Platforms  
### Using Deep Learning and Stochastic Demand Simulation  

**By Abdulrahman Jaber Ageeli – February 2026**

---

## 📌 Project Overview

Food delivery platforms operate in highly dynamic environments where customer demand depends on contextual factors such as distance, time of day, and weather conditions. Static pricing policies ignore this variability and may result in suboptimal revenue outcomes.

This project develops a **context-aware revenue optimization framework** that integrates:

- Stochastic demand simulation  
- Economically consistent price sensitivity modeling  
- Logistic Regression (baseline)  
- Deep Neural Networks (nonlinear modeling)  

The objective is to estimate the probability of order acceptance under varying prices and determine the **revenue-maximizing delivery fee**.

Unlike purely predictive models trained on fixed historical labels, this project constructs a controlled demand simulator that embeds economic structure and nonlinear contextual effects.

---

## 📊 Dataset

We use the Kaggle DoorDash dataset:

`historical_data.csv`

Selected variables:

- `created_at`  
- `estimated_store_to_consumer_driving_duration`  

These variables are transformed into structured contextual features.

---

## ⚙️ Methodology

### 1️⃣ Feature Engineering

#### Distance Estimation

Driving duration is converted into distance (km):

$$
\text{distance} = \frac{\text{duration}_{\text{seconds}}}{3600} \times 40
$$

where 40 km/h is assumed as average urban driving speed.

---

#### Peak Hour Indicator

Peak periods are defined as:

- Lunch: 11:00–14:00  
- Dinner: 18:00–21:00  

A binary variable `is_peak` is created.

---

#### Weather Simulation

Since weather is not available, rain probability is simulated conditionally:

$$
P(\text{rain}) = 0.2 + 0.1 \times \text{is\_peak}
$$

This increases contextual variability during high-demand periods.

---

### 2️⃣ Dynamic Base Pricing Model

Instead of using a fixed delivery fee:

$$
P(\mathrm{rain}) = 0.2 + 0.1 \times \mathrm{is\_peak}
$$

This captures:
- Fixed operational cost (8)
- Distance-based marginal cost

---

### 3️⃣ Data Leakage Prevention

To ensure methodological rigor:

- Train/test split is performed **before counterfactual price expansion**
- Each order context belongs exclusively to one split
- Price perturbations are generated separately within each split

This prevents the same order context from appearing in both training and testing data under different prices.

---

### 4️⃣ Stochastic Demand Simulation

For each order, 10 counterfactual prices are generated:

$$
\text{price} = P_{\text{base}} + \delta
$$

where

$$
\delta \sim \mathcal{U}\left(-0.5 P_{\text{base}}, +0.5 P_{\text{base}}\right)
$$

Minimum price is constrained to \$2.

---

### Nonlinear Price Sensitivity

Price elasticity varies with context:

$$
\alpha(X) =
0.15
+ 0.05 \cdot \text{distance}
+ 0.02 \cdot \text{distance}^2
+ 0.08 \cdot (\text{distance} \times \text{is\_rainy})
- 0.07 \cdot \text{is\_peak}
$$

This introduces:

- Quadratic distance effects  
- Distance–weather interaction  
- Reduced elasticity during peak hours  

---

### Acceptance Probability

First compute:

$$
z = -\alpha(X)\left(\text{price} - P_{\text{base}}\right) + \epsilon
$$

Then acceptance probability:

$$
P(\text{accept}) = \sigma(z)
$$

where

$$
\epsilon \sim \mathcal{N}(0, 0.5)
$$

and $\sigma(\cdot)$ denotes the sigmoid function.

The final acceptance label is drawn stochastically from this probability, producing a nonlinear decision surface.

---

## 🤖 Models

### Logistic Regression
- Linear decision boundary  
- Baseline model  

### Deep Neural Network

Architecture:
- 64 → 32 → 16 hidden units  
- ReLU activations  
- Dropout (0.2)  
- Early stopping  

The DNN captures nonlinear interactions embedded in $\alpha(X)$.

---

## 📈 Evaluation Metrics

- **ROC-AUC** (classification performance)  
- **Brier Score** (probability calibration)  
- **Revenue Optimization Curve**  

Expected revenue:

$$
\text{Revenue} = \text{price} \times P(\text{accept})
$$

---

## 🧠 Results

| Model | AUC | Brier Score |
|-------|------|------------|
| Logistic Regression | ~0.945 | ~0.091 |
| Deep Neural Network | ~0.957 | ~0.084 |

Key Findings:

- DNN outperforms Logistic Regression in both discrimination and calibration.
- Nonlinear contextual effects improve predictive performance.
- Revenue optimization curve identifies a clear revenue-maximizing price.
- No signs of severe overfitting observed.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
