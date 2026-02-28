# 🚚 Dynamic Pricing Using Deep Learning
## Revenue Optimization with Simulated Demand Data

---

# 📌 Project Overview

This project applies **Deep Learning** to a dynamic pricing problem in food delivery systems.

Using historical operational data (`historical_data.csv`), we simulate customer price responses and train machine learning models to estimate order acceptance probability and optimize expected revenue.

The optimization objective is:

$$
\text{Revenue} = \text{Price} \times P(\text{Accept})
$$

Because the original dataset contains only historically observed prices — and not alternative pricing experiments — we construct a structured simulation framework to approximate customer reactions under counterfactual pricing scenarios.

---

# 📊 Dataset Description — `historical_data.csv`

The dataset includes operational features such as:

- Delivery duration (seconds)
- Order timing
- Context variables

## Statistical Limitation of the Dataset

The dataset reflects realized pricing decisions only.  
It does not contain:

- Observations of the same order under multiple price levels  
- Explicit customer price elasticity  
- Counterfactual demand outcomes  

From a statistical perspective, this creates a **missing potential outcomes problem**.  
Without sufficient variation in price, a supervised model cannot learn how demand changes when price changes.

This motivates the use of a structured simulation approach.

---

# 🧠 Methodology

To address the lack of price variation, we build a **stochastic economic simulator** that expands the dataset with controlled price experimentation.

---

## 1️⃣ Feature Engineering

### Distance Approximation

Distance is estimated from delivery duration assuming an average speed of 40 km/h:

$$
\text{Distance} = \left( \frac{\text{Duration}}{3600} \right) \times 40
$$

---

### Base Price Model

A baseline delivery price is defined as:

$$
P_{\text{base}} = 8 + 0.8 \times \text{Distance}
$$

This ensures price increases proportionally with delivery cost.

---

## 2️⃣ Controlled Price Variation

To simulate alternative pricing scenarios:

$$
\text{Price} = P_{\text{base}} + \delta
$$

where:

$$
\delta \sim \text{Uniform}\left(-0.5 P_{\text{base}}, +0.5 P_{\text{base}}\right)
$$

This generates realistic upward and downward price adjustments while maintaining economic plausibility.

---

## 3️⃣ Nonlinear Price Sensitivity

Customer elasticity is modeled as:

$$
\alpha(X) =
0.15
+ 0.05 \cdot \text{Distance}
+ 0.02 \cdot \text{Distance}^2
+ 0.08 \cdot (\text{Distance} \cdot \text{Rainy})
- 0.07 \cdot \text{Peak}
$$

This structure allows elasticity to vary nonlinearly across contexts.

---

## 4️⃣ Acceptance Probability Model

Acceptance probability is defined using a logistic function:

$$
P(\text{Accept}) =
\sigma\left(
- \alpha(X)\left(\text{Price} - P_{\text{base}}\right)
+ \varepsilon
\right)
$$

where:

$$
\varepsilon \sim \mathcal{N}(0, 0.5)
$$

The logistic function is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This introduces stochastic behavior consistent with real-world customer uncertainty.

The simulator generates binary acceptance labels, creating a supervised learning dataset.

---

# 🤖 Model Selection and Justification

Two models are trained and compared.

## Logistic Regression

- Serves as a linear benchmark  
- Provides interpretable baseline results  
- Establishes minimum expected performance  

## Deep Neural Network (DNN)

Architecture: Multi-layer fully connected network.

Justification:

- The pricing function includes nonlinear and interaction effects.
- Logistic Regression cannot capture quadratic and cross-feature interactions effectively.
- A DNN can approximate complex nonlinear demand surfaces.

The architecture balances model capacity with overfitting risk given tabular structured data.

---

# 📈 Results and Interpretation

Models are evaluated using:

- ROC–AUC  
- Brier Score  
- Revenue optimization curve  

### Interpretation

- The DNN achieves higher ROC–AUC, indicating better ranking ability.
- Lower Brier Score suggests improved probability calibration.
- The revenue curve shows a clear interior maximum, confirming that optimal pricing is neither minimal nor maximal.

The performance improvement demonstrates that nonlinear modeling better captures contextual elasticity effects.

---

# 🏗 Project Workflow

1. Load historical dataset  
2. Engineer contextual features  
3. Generate synthetic price variations  
4. Simulate acceptance outcomes  
5. Train ML models  
6. Evaluate predictive performance  
7. Analyze revenue optimization  

---

# ⚠ Limitations

- Weather and demand variation are simulated rather than externally validated.
- The elasticity function is assumed rather than estimated from real experiments.
- The model does not incorporate competition or multi-agent pricing.
- Real-world deployment would require A/B testing validation.

---

# 🔮 Future Work

- Incorporate real experimental pricing data.
- Replace simulated weather with API-based historical weather.
- Explore gradient boosting models for structured tabular data.
- Extend to multi-period dynamic pricing with reinforcement learning.

---

# 🚀 How to Run

1. Install dependencies:
