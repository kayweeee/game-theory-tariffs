import streamlit as st
from graphviz import Digraph
import numpy as np

# === Country Profiles ===
countries = {
    "China": {
        "G": 0.5, "N": 0.7, "D": 0.4, "F": 0.2,
        "GDP": 18, "TD": 0.112,
        "GDP_us": 25, "TD_us": 143.5 / 2100
    },
    "Mexico": {
        "G": 0.6, "N": 0.4, "D": 0.6, "F": 0.9,
        "GDP": 1.79, "TD": 0.446,
        "GDP_us": 25, "TD_us": 334 / 2100
    },
    "EU": {
        "G": 0.7, "N": 0.3, "D": 0.9, "F": 0.7,
        "GDP": 18.59, "TD": 0.206,
        "GDP_us": 25, "TD_us": 370.2 / 2100
    }
}

# === UI ===
st.title("🇺🇸 US Tariff Game Model")

selected_country = st.selectbox("Select Country:", list(countries.keys()))
profile = countries[selected_country]
player2 = selected_country

st.markdown(f"### Belief Parameters for {player2}")
st.write(f"- Trade Openness (G): {profile['G']}")
st.write(f"- Nationalism (N): {profile['N']}")
st.write(f"- Democracy (D): {profile['D']}")
st.write(f"- Friendliness (F): {profile['F']}")

# US political parameters
st.markdown("### US Political Parameters")
alpha = st.number_input("α (Manufacturing gain per % tariff)", value=1.0)
beta = st.number_input("β (Political penalty from retaliation)", value=0.8)
gamma = st.number_input("γ (Prestige bonus)", value=0.2)

# Tariff levels
t1 = 0.0
t2 = 0.3
t3 = 0.6
z1_tariff = 0.6  # moderate second round
z2_tariff = 0.9  # aggressive second round
t_ch_low = 0.15
t_ch_high = 0.3

# Model functions
def softmax(scores):
    e = np.exp(scores - np.max(scores))
    return e / e.sum()

def response_probs(G, N, D, F, T):
    S_R = 1.0 * G + 1.2 * N + 1.5 * T - 1.0 * D - 1.3 * F
    S_N = 1.3 * F + 1.0 * D + 1.2 * (1 - N) + 0.8 * G
    S_P = 1.2 * (1 - G) + 1.0 * D + 1.5 * (1 - T) + 1.0 * F
    return softmax(np.array([S_R, S_N, S_P]))

def us_payoff(t_us, t_ch, retaliated, alpha, beta, gamma, GDP_us, TD_us):
    e_us = -0.31
    damage = GDP_us * (t_ch * TD_us * (1 - e_us)) if retaliated else 0
    political = alpha * t_us - beta * retaliated + gamma
    return -damage + political

def foreign_payoff(t_us, t_ch, GDP, TD):
    e = -0.31
    damage = GDP * (t_us * TD * (1 - e))
    retaliation_cost = 0.1 * damage if t_ch > 0 else 0
    return -damage - retaliation_cost

# Extract country-specific data
G, N, D, F = profile["G"], profile["N"], profile["D"], profile["F"]
GDP_us, TD_us = profile["GDP_us"], profile["TD_us"]
GDP_f, TD_f = profile["GDP"], profile["TD"]

# Terminal nodes: first-stage outcomes
T1 = us_payoff(t1, 0, 0, alpha, beta, gamma, GDP_us, TD_us)
T2 = us_payoff(t3, 0, 0, alpha, beta, gamma, GDP_us, TD_us)
T3 = us_payoff(t3, 0, 0, alpha, beta, gamma, GDP_us, TD_us)
T4 = us_payoff(t3, t_ch_high, 1, alpha, beta, gamma, GDP_us, TD_us)
T5 = us_payoff(t2, 0, 0, alpha, beta, gamma, GDP_us, TD_us)

# US second-stage moves
T6a = us_payoff(z1_tariff, 0, 0, alpha, beta, gamma, GDP_us, TD_us)
T6b = us_payoff(z2_tariff, 0, 0, alpha, beta, gamma, GDP_us, TD_us)
T7a = us_payoff(z1_tariff, t_ch_low, 1, alpha, beta, gamma, GDP_us, TD_us)
T7b = us_payoff(z2_tariff, t_ch_high, 1, alpha, beta, gamma, GDP_us, TD_us)

# Foreign payoffs
C1 = foreign_payoff(t1, 0, GDP_f, TD_f)
C2 = foreign_payoff(t3, 0, GDP_f, TD_f)
C3 = foreign_payoff(t3, 0, GDP_f, TD_f)
C4 = foreign_payoff(t3, t_ch_high, GDP_f, TD_f)
C5 = foreign_payoff(t2, 0, GDP_f, TD_f)
C6a = foreign_payoff(z1_tariff, 0, GDP_f, TD_f)
C6b = foreign_payoff(z2_tariff, 0, GDP_f, TD_f)
C7a = foreign_payoff(z1_tariff, t_ch_low, GDP_f, TD_f)
C7b = foreign_payoff(z2_tariff, t_ch_high, GDP_f, TD_f)

# Response probabilities
probs_x2 = response_probs(G, N, D, F, t2)
probs_x3 = response_probs(G, N, D, F, t3)
p2 = dict(zip(["retaliate", "negotiate", "no_response"], probs_x2))
p3 = dict(zip(["retaliate", "negotiate", "no_response"], probs_x3))

# Expected payoffs (updated for new branching logic)
U_x1 = T1
U_x2 = (
    p2["no_response"] * T5 +
    p2["negotiate"] * max(T6a, T6b) +
    p2["retaliate"] * max(T7a, T7b)
)
U_x3 = (
    p3["no_response"] * T2 +
    p3["negotiate"] * T3 +
    p3["retaliate"] * T4
)

C_x1 = C1
C_x2 = (
    p2["no_response"] * C5 +
    p2["negotiate"] * max(C6a, C6b) +
    p2["retaliate"] * max(C7a, C7b)
)
C_x3 = (
    p3["no_response"] * C2 +
    p3["negotiate"] * C3 +
    p3["retaliate"] * C4
)

# Best strategy determination
us_options = {"x1": U_x1, "x2": U_x2, "x3": U_x3}
china_x2_best = max(p2, key=p2.get)
china_x3_best = max(p3, key=p3.get)
us_best = max(us_options, key=us_options.get)

# Outputs
st.subheader("US Expected Payoffs")
st.write(f"**x1 (No Tariff):** {round(U_x1, 2)}")
st.write(f"**x2 (Low Tariff):** {round(U_x2, 2)}")
st.write(f"**x3 (High Tariff):** {round(U_x3, 2)}")

st.subheader(f"{player2} Expected Payoffs")
st.write(f"**x1:** {round(C_x1, 2)}")
st.write(f"**x2:** {round(C_x2, 2)}")
st.write(f"**x3:** {round(C_x3, 2)}")

st.subheader("Best Responses")
st.markdown(f"- **Best initial move for US:** `{us_best}`")
st.markdown(f"- **{player2}'s likely response to x2:** `{china_x2_best}`")
st.markdown(f"- **{player2}'s likely response to x3:** `{china_x3_best}`")

# === Game Tree Display ===
dot = Digraph()
dot.attr(rankdir='TB', size='20')

# Player nodes
dot.node("US1", "US")
dot.node("P2_x2", player2)
dot.node("P2_x3", player2)
dot.node("US2_y2", "US")
dot.node("US2_y3", "US")

# Terminal outcomes
dot.node("T1", f"(US: {round(T1, 2)}, {player2}: {round(C1, 2)})")
dot.node("T2", f"(US: {round(T2, 2)}, {player2}: {round(C2, 2)})")
dot.node("T3", f"(US: {round(T3, 2)}, {player2}: {round(C3, 2)})")
dot.node("T4", f"(US: {round(T4, 2)}, {player2}: {round(C4, 2)})")
dot.node("T5", f"(US: {round(T5, 2)}, {player2}: {round(C5, 2)})")
dot.node("T6a", f"(US: {round(T6a, 2)}, {player2}: {round(C6a, 2)})")
dot.node("T6b", f"(US: {round(T6b, 2)}, {player2}: {round(C6b, 2)})")
dot.node("T7a", f"(US: {round(T7a, 2)}, {player2}: {round(C7a, 2)})")
dot.node("T7b", f"(US: {round(T7b, 2)}, {player2}: {round(C7b, 2)})")

# Game flow
dot.edge("US1", "T1", label="x1")
dot.edge("US1", "P2_x2", label="x2")
dot.edge("US1", "P2_x3", label="x3")

dot.edge("P2_x2", "T5", label=f"y1 = {p2['no_response']:.2f}")
dot.edge("P2_x2", "US2_y2", label=f"y2 = {p2['negotiate']:.2f}")
dot.edge("P2_x2", "US2_y3", label=f"y3 = {p2['retaliate']:.2f}")

dot.edge("P2_x3", "T2", label=f"y1 = {p3['no_response']:.2f}")
dot.edge("P2_x3", "T3", label=f"y2 = {p3['negotiate']:.2f}")
dot.edge("P2_x3", "T4", label=f"y3 = {p3['retaliate']:.2f}")

dot.edge("US2_y2", "T6a", label="z1 (0.6)")
dot.edge("US2_y2", "T6b", label="z2 (0.9)")
dot.edge("US2_y3", "T7a", label="z1 (0.6)")
dot.edge("US2_y3", "T7b", label="z2 (0.9)")

st.subheader("Game Tree")
st.graphviz_chart(dot)
