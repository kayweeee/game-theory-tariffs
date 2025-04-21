import streamlit as st
from graphviz import Digraph
import numpy as np

# === Country Profiles ===
countries = {
    "China":  {"GDP": 17.79, "Trade": 0.18,  "Nationalism": 8.0, "Friendliness": 0.2},
    "Mexico": {"GDP": 1.79,  "Trade": 0.446, "Nationalism": 4.0, "Friendliness": 0.9},
    "EU":     {"GDP": 18.4,  "Trade": 0.15,  "Nationalism": 2.0, "Friendliness": 0.5},
}

# === Normalisation constants ===
GDP_MIN, GDP_MAX   = 0.3, 25.0
TRADE_MAX          = 0.75
NATIONALISM_MAX    = 10.0

# ------------------------------------------------------------------
#  Utility helpers
# ------------------------------------------------------------------
def softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max())
    return e / e.sum()

def response_type_probs(G, N, D, F):
    """Nature’s soft‑max score for each retaliation type"""
    S_ret  = 1.4*G + 1.5*N - 1.0*D - 1.2*F
    S_low  = 1.2*F + 0.8*D + 0.9*(1-N) + 0.6*G
    S_none = 1.1*(1-G) + 1.0*D + 1.0*F
    return softmax(np.array([S_ret, S_low, S_none]))

def us_payoff_simple(t_us, retaliation, GDP_partner, friendliness,
                     partner_type, gamma: float = 2.5):
    penalty = {"high": .5, "low": .2, "none": .1}[partner_type]
    tariff_gain          = t_us * (1 - friendliness)
    retaliation_cost     = retaliation * GDP_partner * penalty
    friendliness_penalty = t_us * friendliness * gamma
    return round(tariff_gain - retaliation_cost - friendliness_penalty, 2)

def foreign_payoff_simple(t_us, retaliation, GDP, friendliness, partner_type):
    type_sensitivity = {"high": 1.0, "low": .6, "none": .3}
    sens = type_sensitivity[partner_type]

    tariff_damage    =  t_us * GDP * sens
    retaliation_gain =  retaliation * (1 - friendliness) * sens

    gdp_factor       = 1 / (GDP + 0.1)
    base_cost        = retaliation * friendliness * gdp_factor * 5
    tariff_discount  = t_us * 3
    retaliation_cost = max(base_cost - tariff_discount, 0)

    return round(-tariff_damage + retaliation_gain - retaliation_cost, 2)

# ------------------------------------------------------------------
#  Streamlit UI
# ------------------------------------------------------------------
st.title("🇺🇸 Simplified Tariff Game")

# 1. country selector ------------------------------------------------
country = st.selectbox("Select Country:", countries.keys())
prof    = countries[country]

# 2. policy sliders --------------------------------------------------
st.sidebar.header("Policy Settings")
t_low  = st.sidebar.slider("US low‑tariff rate",    0.00, 0.30, 0.10, 0.01)
t_high = st.sidebar.slider("US high‑tariff rate",   0.00, 0.60, 0.40, 0.01)
t_ret  = st.sidebar.slider(f"{country} retaliation rate", 0.00, 0.40, 0.20, 0.01)
gamma  = st.sidebar.slider("Friendliness penalty γ", 0.0, 5.0, 2.5, 0.1)

# 3. normalised state vars ------------------------------------------
G = (prof["GDP"]          - GDP_MIN) / (GDP_MAX - GDP_MIN)
D =  prof["Trade"]        / TRADE_MAX
N =  prof["Nationalism"]  / NATIONALISM_MAX
F =  prof["Friendliness"]

types = ["high", "low", "none"]
p     = dict(zip(types, response_type_probs(G, N, D, F)))

# ------------------------------------------------------------------
#  Payoff matrix for every info set
# ------------------------------------------------------------------
payoffs = {}
for typ in types:
    for us_act, τ_us in (("low", t_low), ("high", t_high)):
        for fr_act, τ_fr in (("no_retaliate", 0.0), ("retaliate", t_ret)):
            key        = f"{typ}_{us_act}_{fr_act}"
            us_pay     = us_payoff_simple(τ_us, τ_fr, prof["GDP"], F, typ, gamma)
            fr_pay     = foreign_payoff_simple(τ_us, τ_fr, prof["GDP"], F, typ)
            payoffs[key] = (us_pay, fr_pay)

# ------------------------------------------------------------------
#  Foreign best response given US action
# ------------------------------------------------------------------
def foreign_best_action(typ: str, us_act: str) -> str:
    r_payoff  = payoffs[f"{typ}_{us_act}_retaliate"][1]
    nr_payoff = payoffs[f"{typ}_{us_act}_no_retaliate"][1]
    return "retaliate" if r_payoff > nr_payoff else "no_retaliate"

# Expected US payoffs (sub‑game‑perfect) ----------------------------
exp_us_low  = sum(p[t] * payoffs[f"{t}_low_{foreign_best_action(t,'low')}"][0]  for t in types)
exp_us_high = sum(p[t] * payoffs[f"{t}_high_{foreign_best_action(t,'high')}"][0] for t in types)
best_action = "high" if exp_us_high > exp_us_low else "low"

# ------------------------------------------------------------------
#  Outputs
# ------------------------------------------------------------------
st.subheader("Nature’s belief over retaliation types")
for t in types:
    st.markdown(f"- **{t.title()}**: {p[t]:.2f}")

st.subheader("Sub‑game‑perfect expected US payoffs")
st.write(f"**Low tariff:**  {exp_us_low:.2f}")
st.write(f"**High tariff:** {exp_us_high:.2f}")
st.success(f"✅ Best US strategy: **{best_action.upper()} TARIFF**")

# ------------------------------------------------------------------
#  Game tree (Graphviz)
# ------------------------------------------------------------------
dot = Digraph()
dot.attr(rankdir="LR", size="12,6")
dot.node("Nature", "Nature")

for typ in types:
    us_node = f"US_{typ}"
    dot.node(us_node, f"US")
    dot.edge("Nature", us_node, label=f"{typ}  p={p[typ]:.2f}")

    for us_act, τ_us in (("low", t_low), ("high", t_high)):
        fr_node = f"{typ}_{us_act}"
        dot.node(fr_node, f"{country} ({typ})")
        dot.edge(us_node, fr_node, label=f"{us_act.title()} τ={τ_us:.2f}")

        for fr_act, τ_fr in (("retaliate", t_ret), ("no_retaliate", 0.0)):
            leaf   = f"{typ}_{us_act}_{fr_act}"
            u_pay, f_pay = payoffs[leaf]
            dot.node(leaf, f"(US:{u_pay}, {country}:{f_pay})")
            dot.edge(fr_node, leaf, label=f"{fr_act.title()}  τ={τ_fr:.2f}")

st.subheader("Game tree")
st.graphviz_chart(dot)
