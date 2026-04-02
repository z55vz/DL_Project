# AI Dynamic Pricing using Deep Learning

## 🧠 Overview
This project builds an AI system that determines the **optimal delivery price** for each order.

Instead of using a fixed price, the system:
- Predicts whether a customer will accept a price
- Evaluates multiple price options
- Selects the price that maximizes revenue

---

## 🎯 Objective

The goal is to maximize:

**Expected Revenue = Price × P(accept)**

Where:
- `Price` = delivery fee  
- `P(accept)` = probability the customer accepts the price  

---

## ⚙️ Methodology

### 1. Feature Engineering
The model uses key variables that influence customer decisions:

- `distance_km`: Delivery distance  
- `is_peak`: Whether the order is during peak hours (0 or 1)  
- `is_rainy`: Simulated weather condition  
- `p_base`: Base delivery price  
- `price`: Tested delivery price  

Base price is calculated as:

p_base = 8 + 0.8 × distance

---

### 2. Market Simulation

Since real acceptance data is unavailable, customer behavior is simulated using a logistic model:

P(accept) = 1 / (1 + e^(-z))

Where:

z = -α(price - p_base) + ε

α depends on:
- distance
- weather
- peak hours

This allows the system to generate realistic training data.

---

### 3. Model Training

Two models are used:

- Logistic Regression (baseline)
- Deep Neural Network (main model)

The neural network captures complex non-linear relationships between price and customer behavior.

---

### 4. Price Optimization

For each request, the model evaluates multiple pricing scenarios and selects:

Optimal Price = argmax (price × P(accept))

This step converts predictions into actual business decisions.

---

## 📊 Sample Output

Example of optimized pricing decisions:

| distance_km | p_base | Optimal_Price | Strategy |
|------------|--------|---------------|----------|
| 3.44       | 10.75  | 8.23          | Demand Stimulation |
| 5.62       | 12.49  | 10.07         | Demand Stimulation |
| 7.38       | 13.91  | 12.34         | Demand Stimulation |
| 8.97       | 15.18  | 13.78         | Demand Stimulation |

---

## 🧾 Column Explanation

- `distance_km`: Delivery distance in kilometers  
- `p_base`: Base price calculated from distance  
- `Optimal_Price`: Price selected by the model  
- `Strategy`: Pricing decision type  

---

## 🧠 Pricing Strategies

### 1. Demand Stimulation
- Price is **lower than base price**
- Used when customers are price-sensitive
- Goal: Increase acceptance rate

### 2. Premium Strategy
- Price is **higher than base price**
- Used when demand is strong
- Goal: Increase profit margin

---

## 📈 Key Results

- Deep Neural Network outperforms Logistic Regression  
- Model predictions are well-calibrated  
- Revenue increases significantly after optimization (≈ 60%)  

---

## 🚀 How to Run

Install dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
