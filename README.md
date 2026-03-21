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
The system extracts key features to understand the delivery context:
* **Distance**: Estimated store-to-consumer distance.
* **Time**: Identification of Peak vs. Off-peak hours.
* **Base Price**: The starting price used for comparison.

### 2. Market Simulation
Since real-world historical data often lacks "what-if" scenarios (counterfactuals), a stochastic simulator generates realistic customer responses based on non-linear demand patterns.

### 3. Model Training
Two models are trained and compared to prove the effectiveness of Deep Learning:
* **Logistic Regression**: Serves as the statistical baseline.
* **Deep Neural Network (DNN)**: The main model featuring Batch Normalization and Dropout for optimized performance.

### 4. Price Optimization
For each request, the system runs an optimization loop that tests a grid of potential prices and selects the one yielding the highest expected revenue.

---

## 📊 Results

### Model Performance
| Model | ROC-AUC | Brier Score |
|------|--------|------------|
| Logistic Regression | ~0.8185 | ~0.17 |
| **Deep Neural Network** | **~0.8189** | **~0.17** |

### Revenue Improvement
The AI system demonstrates a significant improvement in revenue compared to fixed baseline pricing, typically achieving a **~19.86% increase** in optimized scenarios.

---

## 📈 Visual Results

| ROC Curve | Confusion Matrix |
|:---:|:---:|
| ![ROC Curve](Images/لقطة%20شاشة%202026-03-21%20055317.png) | ![Confusion Matrix](Images/لقطة%20شاشة%202026-03-21%20055345.png) |

| Calibration Curve | Revenue Impact |
|:---:|:---:|
| ![Calibration Curve](Images/لقطة%20شاشة%202026-03-21%20055359.png) | ![Revenue Comparison](Images/لقطة%20شاشة%202026-03-21%20055414.png) |

---

## 📋 Sample Output (Top 10 Requests)
| distance_km | is_peak | p_base | Optimal_Price | Exp_Revenue | Strategy |
|-------------|---------|--------|---------------|-------------|----------|
| 3.484 | 0 | 10.788 | 8.853 | 5.916 | Demand Stimulation |
| 1.484 | 0 | 9.188 | 8.554 | 4.638 | Demand Stimulation |
| 7.720 | 1 | 14.176 | 11.243 | 9.122 | Demand Stimulation |

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
