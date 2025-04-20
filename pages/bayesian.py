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

# === Game Parameters ===
mu = 2.0  # US gain per tariff
nu = 1.5  # Foreign gain per retaliation
t2 = 0.1  # US low tariff
t3 = 0.4  # US high tariff
t_ch = 0.2  # retaliation level
penalty_scale = 2.0  # friendliness-based penalty for US

# === Normalize for type probabilities ===
GDP_MIN = 0.3
GDP_MAX = 25.0
TRADE_MAX = 0.75
NATIONALISM_MAX = 10.0

def softmax(scores):
    e = np.exp(scores - np.max(scores))
    return e / e.sum()

def response_type_probs(G, N, D, F):
    S_ret = 1.4 * G + 1.5 * N - 1.0 * D - 1.2 * F
    S_low = 1.2 * F + 0.8 * D + 0.9 * (1 - N) + 0.6 * G
    S_none = 1.1 * (1 - G) + 1.0 * D + 1.0 * F
    return softmax(np.array([S_ret, S_low, S_none]))

def us_payoff(t_us, t_ch, GDP_us, TD_us, mu, ID_us, friendliness, penalty_scale=2.0):
    gain = mu * t_us * ID_us
    retaliation_damage = 1.5 * GDP_us * t_ch * TD_us
    friendly_penalty = -friendliness * penalty_scale if t_us > 0.2 else 0
    return round(gain - retaliation_damage + friendly_penalty, 2)

def foreign_payoff(t_us, t_ch, GDP, TD, nu, ID, type_factor):
    gain = nu * t_ch * ID
    damage = GDP * t_us * TD
    alignment = type_factor  # scalar based on alignment sensitivity
    return round(gain * alignment - damage, 2)

# === Streamlit UI ===
st.title("🇺🇸 US Tariff Game with Friendliness-Based Penalty")

selected_country = st.selectbox("Select Country:", list(countries.keys()))
profile = countries[selected_country]
player2 = selected_country

# Normalized inputs
G = (profile["GDP"] - GDP_MIN) / (GDP_MAX - GDP_MIN)
D = profile["Trade"] / TRADE_MAX
N = profile["Nationalism"] / NATIONALISM_MAX
F = profile["Friendliness"]

type_probs = response_type_probs(G, N, D, F)
types = ["high", "low", "none"]
p = dict(zip(types, type_probs))
GDP_us, TD_us = profile["GDP_us"], profile["TD_us"]
GDP_f, TD_f = profile["GDP"], profile["Trade"]
ID_us, ID_f = profile["ID_us"], profile["ID"]

# === Payoff matrix calculation ===
type_pref = {"high": "retaliate", "low": "no_retaliate", "none": "no_retaliate"}
payoffs = {}

type_alignment_bonus = {
    "high": {"retaliate": 1.2, "no_retaliate": 0.8},
    "low": {"retaliate": 1.0, "no_retaliate": 1.0},
    "none": {"retaliate": 0.8, "no_retaliate": 1.2},
}

for typ in types:
    for x, t_us in zip(["high", "low"], [t3, t2]):
        for y, t_chosen in zip(["retaliate", "no_retaliate"], [t_ch, 0.0]):
            node = f"{typ}_{x}_{y}"
            us = us_payoff(
                t_us, t_chosen, GDP_us, TD_us, mu, ID_us,
                friendliness=F
            )
            ch = foreign_payoff(
                t_us, t_chosen, GDP_f, TD_f, nu, ID_f,
                type_factor=type_alignment_bonus[typ][y]
            )
            payoffs[node] = (us, ch)

# === US Expected Payoff ===
U_high = sum([p[typ] * payoffs[f"{typ}_high_retaliate"][0] * 0.5 + p[typ] * payoffs[f"{typ}_high_no_retaliate"][0] * 0.5 for typ in types])
U_low = sum([p[typ] * payoffs[f"{typ}_low_retaliate"][0] * 0.5 + p[typ] * payoffs[f"{typ}_low_no_retaliate"][0] * 0.5 for typ in types])
us_options = {"High Tariff": U_high, "Low Tariff": U_low}
us_best = max(us_options, key=us_options.get)

# Display Results
st.subheader("Type Probabilities (Drawn by Nature)")
for k in types:
    st.markdown(f"- **{k.title()} Type:** {p[k]:.2f}")

st.subheader("US Expected Payoffs")
st.write(f"**High Tariff:** {round(U_high, 2)}")
st.write(f"**Low Tariff:** {round(U_low, 2)}")
st.markdown(f"### 🎯 Best Strategy for US: `{us_best}`")

# === Game Tree Visualization ===
dot = Digraph()
dot.attr(rankdir='LR', size='12,6')
dot.node("Nature", "Nature")

for typ in types:
    us_node = f"US_{typ}"
    dot.node(us_node, "US")
    dot.edge("Nature", us_node, label=f"{typ} ({p[typ]:.2f})")

    for us_action in ["high", "low"]:
        ch_node = f"{typ}_{us_action}"
        dot.node(ch_node, f"{player2} ({typ})")
        dot.edge(us_node, ch_node, label=f"{us_action.title()} Tariff")

        for ch_action in ["retaliate", "no_retaliate"]:
            key = f"{typ}_{us_action}_{ch_action}"
            payoff = payoffs[key]
            dot.node(key, f"(US: {payoff[0]}, {player2}: {payoff[1]})")
            dot.edge(ch_node, key, label=ch_action.title())

st.subheader("Game Tree")
st.graphviz_chart(dot)
