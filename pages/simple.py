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
    },
    "Mexico": {
        "GDP": 1.79,
        "Trade": 0.446,
        "Nationalism": 4.0,
        "Friendliness": 0.7,
    },
    "EU": {
        "GDP": 18.4,
        "Trade": 0.15,
        "Nationalism": 2.0,
        "Friendliness": 0.5,
    }
}

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

def us_payoff_simple(t_us, retaliation, GDP_partner, friendliness, partner_type):
    penalty = {"high": 0.5, "low": 0.2, "none": 0.1}[partner_type]
    gamma = 2.5  # friendliness penalty weight
    tariff_gain = t_us * (1 - friendliness)
    retaliation_cost = retaliation * GDP_partner * penalty
    friendliness_penalty = t_us * friendliness * gamma
    return round(tariff_gain - retaliation_cost - friendliness_penalty, 2)


def foreign_payoff_simple(t_us, retaliation, GDP, friendliness, partner_type):
    type_sensitivity = {"high": 1.0, "low": 0.6, "none": 0.3}
    sens = type_sensitivity[partner_type]
    tariff_damage = t_us * GDP * sens
    retaliation_gain = retaliation * (1 - friendliness) * sens
    return round(-tariff_damage + retaliation_gain, 2)

# === Streamlit UI ===
st.title("🇺🇸 Simplified Tariff Game (GDP + Friendliness + Type-Aware)")

selected_country = st.selectbox("Select Country:", list(countries.keys()))
profile = countries[selected_country]

# Normalized inputs
G = (profile["GDP"] - GDP_MIN) / (GDP_MAX - GDP_MIN)
D = profile["Trade"] / TRADE_MAX
N = profile["Nationalism"] / NATIONALISM_MAX
F = profile["Friendliness"]

# Type probabilities from Nature
types = ["high", "low", "none"]
p = dict(zip(types, response_type_probs(G, N, D, F)))
GDP_f = profile["GDP"]
friendliness = profile["Friendliness"]

# Tariff and retaliation values
t_low = 0.1
t_high = 0.4
retaliate = 0.2
no_retaliate = 0.0

# === Payoff matrix and game tree ===
payoffs = {}
for typ in types:
    for us_action, t_us in zip(["low", "high"], [t_low, t_high]):
        for ch_action, t_ch in zip(["no_retaliate", "retaliate"], [no_retaliate, retaliate]):
            key = f"{typ}_{us_action}_{ch_action}"
            us_val = us_payoff_simple(t_us, t_ch, GDP_f, friendliness, typ)
            ch_val = foreign_payoff_simple(t_us, t_ch, GDP_f, friendliness, typ)
            payoffs[key] = (us_val, ch_val)

# === Game Tree Visualization ===
dot = Digraph()
dot.attr(rankdir='LR', size='12,6')
dot.node("Nature", "Nature")

for typ in types:
    us_node = f"US_{typ}"
    dot.node(us_node, "US")
    dot.edge("Nature", us_node, label=f"{typ} ({p[typ]:.2f})")

    for us_action in ["low", "high"]:
        ch_node = f"{typ}_{us_action}"
        dot.node(ch_node, f"{selected_country} ({typ})")
        dot.edge(us_node, ch_node, label=f"{us_action.title()} Tariff")

        for ch_action in ["retaliate", "no_retaliate"]:
            key = f"{typ}_{us_action}_{ch_action}"
            payoff = payoffs[key]
            dot.node(key, f"(US: {payoff[0]}, {selected_country}: {payoff[1]})")
            dot.edge(ch_node, key, label=ch_action.title())

# === Expected Payoffs for US ===
expected_us_low = sum([p[t] * payoffs[f"{t}_low_retaliate"][0] if payoffs[f"{t}_low_retaliate"][0] > payoffs[f"{t}_low_no_retaliate"][0] else p[t] * payoffs[f"{t}_low_no_retaliate"][0] for t in types])
expected_us_high = sum([p[t] * payoffs[f"{t}_high_retaliate"][0] if payoffs[f"{t}_high_retaliate"][0] > payoffs[f"{t}_high_no_retaliate"][0] else p[t] * payoffs[f"{t}_high_no_retaliate"][0] for t in types])
best_action = "high" if expected_us_high > expected_us_low else "low"

# === Streamlit Outputs ===
st.subheader("Nature's Belief Over Country Types")
for t in types:
    st.markdown(f"- **{t.title()} Type:** {p[t]:.2f}")

st.subheader("Expected US Payoffs")
st.write(f"**Low Tariff:** {round(expected_us_low, 2)}")
st.write(f"**High Tariff:** {round(expected_us_high, 2)}")

st.markdown(f"### ✅ Best Strategy for US: `{best_action.upper()} TARIFF`")

st.subheader("Game Tree")
st.graphviz_chart(dot)
