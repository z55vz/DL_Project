# Dynamic Pricing Using Deep Learning  
### Revenue Optimization with Simulated Demand Data  

---

# 📌 Project Overview

This project applies **Deep Learning** to a dynamic pricing problem in food delivery systems.

Using historical delivery data (`historical_data.csv`), we simulate customer price behavior and train machine learning models to predict order acceptance and optimize revenue.

The objective of the system is:

$$
Revenue = Price \times P(Accept)
$$

Because historical datasets typically contain only observed prices (without alternative price experimentation), we construct a structured simulation framework to generate economically consistent training data.

---

# 📊 Dataset Description — `historical_data.csv`

The dataset contains real delivery information such as:

- Delivery duration (in seconds)
- Order timing indicators
- Contextual operational features

## Core Limitation of the Raw Data

The dataset reflects historical pricing decisions only.

It does **not** include:

- Customer responses to alternative prices  
- Counterfactual pricing scenarios  
- Directly observed price elasticity  

Without controlled variation in price, a supervised learning model cannot learn how demand reacts to price changes.

---

# 🧠 Methodology

To address this limitation, we build a **stochastic economic simulator** that expands the dataset.

---

## 1️⃣ Distance Estimation

Delivery distance is approximated using average speed:

$$
Distance = \frac{Duration\ (seconds)}{3600} \times 40
$$

---

## 2️⃣ Base Price Construction

A baseline delivery price is defined as:

$$
P_{base} = 8 + 0.8 \times Distance
$$

---

## 3️⃣ Price Variation Generation

To simulate alternative pricing scenarios:

$$
Price = P_{base} + \delta
$$

where

$$
\delta \sim \mathcal{U}(-0.5 P_{base},\ +0.5 P_{base})
$$

This creates controlled and realistic price experimentation.

---

## 4️⃣ Nonlinear Price Sensitivity

Customer price sensitivity is modeled as a nonlinear function:

$$
\alpha(X) =
0.15
+ 0.05 \cdot Distance
+ 0.02 \cdot Distance^2
+ 0.08 \cdot (Distance \times Rainy)
- 0.07 \cdot Peak
$$

This allows elasticity to vary across delivery conditions.

---

## 5️⃣ Acceptance Probability

The probability of order acceptance is defined as:

$$
P(Accept) =
\sigma\left(
-\alpha(X)(Price - P_{base}) + \epsilon
\right)
$$

where

$$
\epsilon \sim \mathcal{N}(0, 0.5)
$$

and $ \sigma(\cdot) $ is the logistic function.

This produces a nonlinear and economically coherent demand surface suitable for supervised learning.

---

# 🤖 Models

Two models are trained and compared.

---

## 1️⃣ Logistic Regression

- Linear probability baseline  
- Provides interpretability  
- Establishes benchmark performance  

---

## 2️⃣ Deep Neural Network (DNN)

- Captures nonlinear relationships  
- Models interaction effects  
- Learns complex demand behavior  

The DNN demonstrates improved predictive performance and more accurate revenue estimation compared to Logistic Regression.

---

# 📈 Evaluation Metrics

Models are evaluated using:

- ROC–AUC  
- Brier Score  
- Revenue optimization analysis  

After training, the system can:

- Predict acceptance probability for any given price  
- Generate revenue curves  
- Identify revenue-maximizing pricing levels  

---

# 🏗 Project Workflow

1. Load historical dataset  
2. Engineer contextual features  
3. Generate synthetic price variations  
4. Simulate acceptance behavior  
5. Train machine learning models  
6. Evaluate predictive performance  
7. Analyze revenue optimization  

---

# 🎯 Key Contributions

- Application of Deep Learning to dynamic pricing  
- Construction of a nonlinear economic demand simulator  
- Controlled synthetic data generation for supervised learning  
- Revenue-based model evaluation  

---

# 🚀 How to Run

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
