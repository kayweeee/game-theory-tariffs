import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Tariff Response Calculator", layout="centered")
st.title("Tariff Response Probability Calculator")

st.markdown("""
Select a country to auto-fill economic and geopolitical data, then adjust the sliders to explore how the response probabilities change:
- **Retaliate**
- **Negotiate**
- **Do nothing**
""")

# --- Normalization bounds ---
GDP_MIN = 0.3
GDP_MAX = 25.0
TRADE_MAX = 0.75
NATIONALISM_MAX = 10.0
TARIFF_MAX = 50.0

# --- Preset country values ---
presets = {
    "China": {
        "GDP": 17.79,
        "Trade": 0.18,
        "Nationalism": 8.0,
        "Friendliness": 0.2,
        "Tariff": 25.0
    },
    "EU": {
        "GDP": 18.4,
        "Trade": 0.15,
        "Nationalism": 2.0,
        "Friendliness": 0.7,
        "Tariff": 25.0
    },
    "Canada": {
        "GDP": 2.2,
        "Trade": 0.75,
        "Nationalism": 3.0,
        "Friendliness": 0.9,
        "Tariff": 25.0
    },
    "ASEAN (average)": {
        "GDP": 3.6,
        "Trade": 0.25,
        "Nationalism": 4.0,
        "Friendliness": 0.6,
        "Tariff": 25.0
    },
    "Custom": {
        "GDP": 5.0,
        "Trade": 0.3,
        "Nationalism": 5.0,
        "Friendliness": 0.5,
        "Tariff": 20.0
    }
}

# --- Country selector ---
selected_country = st.selectbox("Choose a country:", presets.keys())

# --- Load defaults from preset ---
default = presets[selected_country]

# --- User-adjustable inputs ---
gdp = st.number_input("GDP (in Trillions USD)", min_value=0.1, value=default["GDP"], step=0.1)
trade_dependence = st.number_input("Exports to U.S. (% of total exports)", min_value=0.0, max_value=100.0,
                                   value=default["Trade"] * 100, step=0.5) / 100
nationalism = st.slider("Nationalism Index (0–10)", 0.0, 10.0, default["Nationalism"], 0.1)
friendliness = st.slider("Friendliness with U.S. (0 = hostile, 1 = ally)", 0.0, 1.0, default["Friendliness"], 0.01)
tariff = st.slider("Tariff Severity Imposed by U.S. (%)", 0.0, 50.0, default["Tariff"], 1.0)

# --- Normalize values ---
G = (gdp - GDP_MIN) / (GDP_MAX - GDP_MIN)
D = trade_dependence / TRADE_MAX
N = nationalism / NATIONALISM_MAX
F = friendliness
T = tariff / TARIFF_MAX

# --- Coefficients ---
alpha = [1.0, 1.2, 1.5, 1.0, 1.3]
beta = [1.3, 1.0, 1.2, 0.8]
gamma = [1.2, 1.0, 1.5, 1.0]

# --- Score calculations ---
S_ret = alpha[0]*G + alpha[1]*N + alpha[2]*T - alpha[3]*D - alpha[4]*F
S_neg = beta[0]*F + beta[1]*D + beta[2]*(1 - N) + beta[3]*G
S_none = gamma[0]*(1 - G) + gamma[1]*D + gamma[2]*(1 - T) + gamma[3]*F

# --- Softmax probabilities ---
exps = np.exp([S_ret, S_neg, S_none])
probs = exps / np.sum(exps)

# --- Display results ---
st.subheader("Response Probabilities")
results = pd.DataFrame({
    "Response": ["Retaliate", "Negotiate", "No Response"],
    "Probability": [round(probs[0], 3), round(probs[1], 3), round(probs[2], 3)]
})
st.table(results)

best_action = results.loc[probs.argmax(), "Response"]
st.success(f"Most likely action: **{best_action}**")

with st.expander("📘 How We Calculate the Probabilities", expanded=False):
    st.markdown("""
We use a simplified game-theoretic model to estimate how likely a country is to respond to a U.S.-imposed tariff in one of three ways:

### 🎯 The Three Response Options:
- **Retaliate** (e.g., impose tariffs in return)
- **Negotiate** (diplomatic or WTO-based approach)
- **No Response** (accept the tariff without action)

### 🔢 Step 1: Real-World Inputs
We take five factors as inputs:
- **GDP** (proxy for economic resilience)
- **Trade dependence on the U.S.**
- **Nationalism or protectionist tendency**
- **Diplomatic friendliness with the U.S.**
- **Tariff severity imposed by the U.S.**

These are normalized to values between 0 and 1:
- GDP is normalized from a min of \$0.3T to a max of \$25T
- Trade dependence from 0% to 75%
- Nationalism from 0 to 10
- Tariff severity from 0% to 50%

### 🧩 Variable Definitions

| Symbol | Description |
|--------|-------------|
| `G` | Economic resilience of the country (normalized GDP) |
| `D` | Trade dependence on the U.S. (normalized export share) |
| `N` | Nationalism / protectionist tendency (normalized from index) |
| `F` | Friendliness with the U.S. (higher = more friendly) |
| `T` | Severity of tariff imposed by the U.S. (normalized) |

                                
### 🧮 Step 2: Score Calculation
Each action is assigned a score based on weighted inputs:
""")

    st.latex(r"S_{ret} = 1.0 \cdot G + 1.2 \cdot N + 1.5 \cdot T - 1.0 \cdot D - 1.3 \cdot F")
    st.latex(r"S_{neg} = 1.3 \cdot F + 1.0 \cdot D + 1.2 \cdot (1 - N) + 0.8 \cdot G")
    st.latex(r"S_{none} = 1.2 \cdot (1 - G) + 1.0 \cdot D + 1.5 \cdot (1 - T) + 1.0 \cdot F")

    st.markdown("""
### 📊 Step 3: Softmax Probability Function
The scores are converted into probabilities using the **softmax function**, which ensures all three probabilities sum to 1:
""")

    st.latex(r"P_i = \frac{e^{S_i}}{e^{S_{ret}} + e^{S_{neg}} + e^{S_{none}}}")

    st.markdown("""
### 🧠 Interpretation
- A higher **nationalism index** increases the chance of **retaliation**
- A higher **friendliness** increases the chance of **negotiation**
- A smaller **GDP** or low **tariff severity** increases the chance of **no response**

This model is not deterministic, but helps simulate the strategic tendencies based on real-world inputs.
""")
