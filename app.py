import streamlit as st
from graphviz import Digraph
import numpy as np

# === Country Profiles ===
countries = {
    "China": {
        "GDP": 17.79,
        "Trade": 0.18,
        "Nationalism": 8.0,
        "Friendliness": 0.2,
        "GDP_us": 27.72,
        "TD_us": 143.5 / 2100,
        "ID_us": 0.114,
        "ID": 0.095
    },
    "Mexico": {
        "GDP": 1.79,
        "Trade": 0.446,
        "Nationalism": 4.0,
        "Friendliness": 0.5,
        "GDP_us": 27.72,
        "TD_us": 334 / 2100,
        "ID_us": 0.156,
        "ID": 0.665
    },
    "EU": {
        "GDP": 18.4,
        "Trade": 0.15,
        "Nationalism": 2.0,
        "Friendliness": 0.7,
        "GDP_us": 27.72,
        "TD_us": 370.2 / 2100,
        "ID_us": 0.191,
        "ID": 0.168
    }
}

# === Normalization Bounds ===
GDP_MIN = 0.3
GDP_MAX = 25.0
TRADE_MAX = 0.75
NATIONALISM_MAX = 10.0
TARIFF_MAX = 100

# === Response Probability Function ===
def softmax(scores):
    e = np.exp(scores - np.max(scores))
    return e / e.sum()

def updated_response_probs(G, N, D, F, T):
    S_ret = 1.0 * G + 1.2 * N + 1.5 * T - 1.0 * D - 1.3 * F
    S_neg = 1.3 * F + 1.0 * D + 1.2 * (1 - N) + 0.8 * G
    S_none = 1.2 * (1 - G) + 1.0 * D + 1.5 * (1 - T) + 1.0 * F
    return softmax(np.array([S_ret, S_neg, S_none]))

# === Payoff Functions (Economic Only) ===
def us_payoff(t_us, t_ch, retaliated, GDP_us, TD_us, mu, ID_us):
    damage = GDP_us * (t_ch * TD_us) if retaliated else 0
    gain = mu * t_us * ID_us
    return -damage + gain

def foreign_payoff(t_us, t_ch, GDP, TD, nu, ID):
    damage = GDP * (t_us * TD)
    gain = nu * t_ch * ID if t_ch > 0 else 0
    return -damage + gain

# === Streamlit UI ===
st.title("🇺🇸 US Tariff Game Model (Economic Focus)")

selected_country = st.selectbox("Select Country:", list(countries.keys()))
profile = countries[selected_country]
player2 = selected_country

st.markdown("""
### 🎮 Action Key

- **x₁**, **x₂**, **x₃** → US initial tariff choices:
  - `x₁` = No tariff  
  - `x₂` = Low tariff  
  - `x₃` = High tariff

- **y₁**, **y₂**, **y₃** → Foreign country's (Player 2) response to US tariff:
  - `y₁` = No response  
  - `y₂` = low_retaliate  
  - `y₃` = Retaliate

- **z₁**, **z₂** → US second move **(if allowed)** after negotiation or retaliation:
  - `z₁` = Moderate escalation (e.g. raise to 60%)  
  - `z₂` = Aggressive escalation (e.g. raise to 90%)

*Note: Number = type of action, Letter = stage of decision or level of player.*
""")

# Fixed economic gain parameters
mu = 0.8
nu = 0.5

# Tariff sliders
st.markdown("### Tariff Parameters (User Controlled)")
t1 = 0.0
t2 = st.slider("x₂: US Low Tariff (%)", 0.1, 0.5, 0.3, step=0.05)
t3 = st.slider("x₃: US High Tariff (%)", 0.3, 1.0, 0.6, step=0.05)
z1_tariff = st.slider("z₁: US Moderate Second Tariff (%)", 0.3, 0.8, 0.6, step=0.05)
z2_tariff = st.slider("z₂: US Aggressive Second Tariff (%)", 0.6, 1.0, 0.9, step=0.05)
t_ch_low = st.slider("y₃: Foreign Retaliation to x₂ (%)", 0.0, 0.5, 0.15, step=0.05)
t_ch_high = st.slider("y₃: Foreign Retaliation to x₃ or z₂ (%)", 0.0, 0.6, 0.3, step=0.05)

# Tariff Levels
# t1 = 0.0
# t2 = 0.3
# t3 = 0.6
# z1_tariff = 0.6
# z2_tariff = 0.9
# t_ch_low = 0.15
# t_ch_high = 0.3

# Normalize inputs
G = (profile["GDP"] - GDP_MIN) / (GDP_MAX - GDP_MIN)
D = profile["Trade"] / TRADE_MAX
N = profile["Nationalism"] / NATIONALISM_MAX
F = profile["Friendliness"]
T_norm_2 = t2 * 100 / TARIFF_MAX
T_norm_3 = t3 * 100 / TARIFF_MAX

# Response probabilities
probs_x2 = updated_response_probs(G, N, D, F, T_norm_2)
probs_x3 = updated_response_probs(G, N, D, F, T_norm_3)
p2 = dict(zip(["high_retaliate", "low_retaliate", "no_response"], probs_x2))
p3 = dict(zip(["high_retaliate", "low_retaliate", "no_response"], probs_x3))

# Inputs
GDP_us, TD_us = profile["GDP_us"], profile["TD_us"]
GDP_f, TD_f = profile["GDP"], profile["Trade"]
ID_us, ID_f = profile["ID_us"], profile["ID"]

# Terminal Payoffs
T1 = us_payoff(t1, 0, 0, GDP_us, TD_us, mu, ID_us)
T2 = us_payoff(t3, 0, 0, GDP_us, TD_us, mu, ID_us)
T3 = us_payoff(t3, 0, 0, GDP_us, TD_us, mu, ID_us)
T4 = us_payoff(t3, t_ch_high, 1, GDP_us, TD_us, mu, ID_us)
T5 = us_payoff(t2, 0, 0, GDP_us, TD_us, mu, ID_us)
T6a = us_payoff(z1_tariff, 0, 0, GDP_us, TD_us, mu, ID_us)
T6b = us_payoff(z2_tariff, 0, 0, GDP_us, TD_us, mu, ID_us)
T7a = us_payoff(z1_tariff, t_ch_low, 1, GDP_us, TD_us, mu, ID_us)
T7b = us_payoff(z2_tariff, t_ch_high, 1, GDP_us, TD_us, mu, ID_us)

C1 = foreign_payoff(t1, 0, GDP_f, TD_f, nu, ID_f)
C2 = foreign_payoff(t3, 0, GDP_f, TD_f, nu, ID_f)
C3 = foreign_payoff(t3, 0, GDP_f, TD_f, nu, ID_f)
C4 = foreign_payoff(t3, t_ch_high, GDP_f, TD_f, nu, ID_f)
C5 = foreign_payoff(t2, 0, GDP_f, TD_f, nu, ID_f)
C6a = foreign_payoff(z1_tariff, 0, GDP_f, TD_f, nu, ID_f)
C6b = foreign_payoff(z2_tariff, 0, GDP_f, TD_f, nu, ID_f)
C7a = foreign_payoff(z1_tariff, t_ch_low, GDP_f, TD_f, nu, ID_f)
C7b = foreign_payoff(z2_tariff, t_ch_high, GDP_f, TD_f, nu, ID_f)

# Expected Payoffs
U_x1 = T1
U_x2 = p2["no_response"] * T5 + p2["low_retaliate"] * max(T6a, T6b) + p2["high_retaliate"] * max(T7a, T7b)
U_x3 = p3["no_response"] * T2 + p3["low_retaliate"] * T3 + p3["high_retaliate"] * T4

# C_x1 = C1
# C_x2 = p2["no_response"] * C5 + p2["low_retaliate"] * max(C6a, C6b) + p2["high_retaliate"] * max(C7a, C7b)
# C_x3 = p3["no_response"] * C2 + p3["low_retaliate"] * C3 + p3["high_retaliate"] * C4

# Best responses
us_options = {"x1": U_x1, "x2": U_x2, "x3": U_x3}
china_x2_best = max(p2, key=p2.get)
china_x3_best = max(p3, key=p3.get)
us_best = max(us_options, key=us_options.get)

# Outputs
st.subheader("US Expected Payoffs")
st.write(f"**x1 (No Tariff):** {round(U_x1, 2)}")
st.write(f"**x2 (Low Tariff):** {round(U_x2, 2)}")
st.write(f"**x3 (High Tariff):** {round(U_x3, 2)}")

# st.subheader(f"{player2} Expected Payoffs")
# st.write(f"**x1:** {round(C_x1, 2)}")
# st.write(f"**x2:** {round(C_x2, 2)}")
# st.write(f"**x3:** {round(C_x3, 2)}")

st.subheader("Best Responses")
st.markdown(f"- **Best initial move for US:** `{us_best}`")
st.markdown(f"- **{player2}'s likely response to x2:** `{china_x2_best}`")
st.markdown(f"- **{player2}'s likely response to x3:** `{china_x3_best}`")

# Game Tree

dot = Digraph()
dot.attr(rankdir='TB', size='20')

dot.node("US1", "US")
dot.node("P2_x2", player2)
dot.node("P2_x3", player2)
dot.node("US2_y2", "US")
dot.node("US2_y3", "US")

dot.node("T1", f"(US: {round(T1, 2)}, {player2}: {round(C1, 2)})")
dot.node("T2", f"(US: {round(T2, 2)}, {player2}: {round(C2, 2)})")
dot.node("T3", f"(US: {round(T3, 2)}, {player2}: {round(C3, 2)})")
dot.node("T4", f"(US: {round(T4, 2)}, {player2}: {round(C4, 2)})")
dot.node("T5", f"(US: {round(T5, 2)}, {player2}: {round(C5, 2)})")
dot.node("T6a", f"(US: {round(T6a, 2)}, {player2}: {round(C6a, 2)})")
dot.node("T6b", f"(US: {round(T6b, 2)}, {player2}: {round(C6b, 2)})")
dot.node("T7a", f"(US: {round(T7a, 2)}, {player2}: {round(C7a, 2)})")
dot.node("T7b", f"(US: {round(T7b, 2)}, {player2}: {round(C7b, 2)})")

dot.edge("US1", "T1", label="x1")
dot.edge("US1", "P2_x2", label="x2")
dot.edge("US1", "P2_x3", label="x3")

dot.edge("P2_x2", "T5", label=f"y1 = {p2['no_response']:.2f}")
dot.edge("P2_x2", "US2_y2", label=f"y2 = {p2['low_retaliate']:.2f}")
dot.edge("P2_x2", "US2_y3", label=f"y3 = {p2['high_retaliate']:.2f}")

dot.edge("P2_x3", "T2", label=f"y1 = {p3['no_response']:.2f}")
dot.edge("P2_x3", "T3", label=f"y2 = {p3['low_retaliate']:.2f}")
dot.edge("P2_x3", "T4", label=f"y3 = {p3['high_retaliate']:.2f}")

dot.edge("US2_y2", "T6a", label="z1")
dot.edge("US2_y2", "T6b", label="z2")
dot.edge("US2_y3", "T7a", label="z1")
dot.edge("US2_y3", "T7b", label="z2")

st.subheader("Game Tree")
st.graphviz_chart(dot)
