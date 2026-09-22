import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

# ── Resolve all paths relative to THIS file's location ────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "results_with_tiers.csv")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ── Load Model & Data ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ model.pkl not found at:\n`{MODEL_PATH}`\n\n"
            "**Fix:** Copy `dashboard/model.pkl` into the same folder as `app.py`, "
            "then restart the app."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            f"❌ results_with_tiers.csv not found at:\n`{DATA_PATH}`\n\n"
            "**Fix:** Run the notebook fully (Task 5 saves this file), "
            "then copy it into the same folder as `app.py`."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)

model, features = load_model()
df = load_data()

TIER_COLORS = {
    "Critical Risk": "#e74c3c",
    "Suspicious":    "#f39c12",
    "Clear":         "#2ecc71"
}

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.title("🛡️ Fraud Ops Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Transaction Explorer", "⚠️ Risk Analysis"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model loaded from:\n`{MODEL_PATH}`")
st.sidebar.caption(f"Data loaded from:\n`{DATA_PATH}`")

# ═══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Detection — Overview")
    st.markdown("---")

    total     = len(df)
    fraud_ct  = int(df["actual_fraud"].sum()) if "actual_fraud" in df.columns else 0
    fraud_rt  = fraud_ct / total * 100 if total else 0

    critical  = df[df["risk_tier"] == "Critical Risk"]
    detect_rt = (critical["actual_fraud"].mean() * 100
                 if "actual_fraud" in critical.columns and len(critical) > 0 else 0)

    amt_col = "TransactionAmt_est" if "TransactionAmt_est" in df.columns else None
    if amt_col and "actual_fraud" in df.columns:
        avg_fraud_amt = df[df["actual_fraud"] == 1][amt_col].mean()
    else:
        avg_fraud_amt = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions",       f"{total:,}")
    col2.metric("Total Fraud Count",        f"{fraud_ct:,}")
    col3.metric("Overall Fraud Rate",       f"{fraud_rt:.2f}%")
    col4.metric("Avg Fraud Amount",         f"${avg_fraud_amt:,.2f}")

    st.markdown("### Risk Tier Distribution")
    tier_counts = df["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Count"]
    fig = px.bar(
        tier_counts, x="Risk Tier", y="Count",
        color="Risk Tier", color_discrete_map=TIER_COLORS,
        title="Transactions per Risk Tier"
    )
    st.plotly_chart(fig, use_container_width=True)

    if "hour" in df.columns:
        st.markdown("### Transaction Volume by Hour of Day")
        hourly = df.groupby(["hour", "risk_tier"]).size().reset_index(name="count")
        fig2 = px.line(
            hourly, x="hour", y="count", color="risk_tier",
            color_discrete_map=TIER_COLORS,
            title="Transactions by Hour of Day"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Fraud Probability Distribution")
    fig3 = px.histogram(
        df, x="fraud_prob", color="risk_tier",
        color_discrete_map=TIER_COLORS, nbins=50,
        barmode="overlay", opacity=0.7,
        title="Distribution of Fraud Probabilities"
    )
    fig3.add_vline(x=0.40, line_dash="dash", line_color="orange",
                   annotation_text="Suspicious threshold (0.40)")
    fig3.add_vline(x=0.75, line_dash="dash", line_color="red",
                   annotation_text="Critical threshold (0.75)")
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION EXPLORER
# ═══════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    tier_filter = col1.multiselect(
        "Filter by Risk Tier",
        ["Critical Risk", "Suspicious", "Clear"],
        default=["Critical Risk", "Suspicious", "Clear"]
    )
    min_prob = col2.slider("Min Fraud Probability", 0.0, 1.0, 0.0, 0.01)
    max_rows = col3.selectbox("Max rows to display", [100, 250, 500, 1000], index=1)

    filtered = df[
        df["risk_tier"].isin(tier_filter) &
        (df["fraud_prob"] >= min_prob)
    ].sort_values("fraud_prob", ascending=False)

    st.write(f"Showing **{min(len(filtered), max_rows):,}** of **{len(filtered):,}** transactions")

    # Pick sensible display columns (whatever exists)
    prefer = ["fraud_prob", "risk_tier", "actual_fraud",
              "TransactionAmt_est", "hour", "is_night", "amt_is_round"]
    display_cols = [c for c in prefer if c in filtered.columns]
    if not display_cols:
        display_cols = list(filtered.columns[:8])

    st.dataframe(
        filtered[display_cols]
        .head(max_rows)
        .style.background_gradient(subset=["fraud_prob"], cmap="Reds")
        .format({"fraud_prob": "{:.4f}",
                 "TransactionAmt_est": "${:,.2f}"}
                if "TransactionAmt_est" in display_cols else {"fraud_prob": "{:.4f}"}),
        use_container_width=True,
        height=420
    )

    st.markdown("---")
    st.markdown("### 🔎 Live Risk Score by Row Index")
    txn_input = st.text_input("Enter row index (0 to {})".format(len(df) - 1), "")
    if txn_input.strip():
        try:
            idx = int(txn_input.strip())
            if 0 <= idx < len(df):
                row   = df.iloc[idx]
                prob  = row["fraud_prob"]
                tier  = row["risk_tier"]
                color = TIER_COLORS.get(tier, "gray")
                st.markdown(
                    f"""<div style='background:{color};padding:20px;
                    border-radius:12px;color:white;font-size:16px;'>
                    <b>Row {idx}</b><br>
                    Fraud Probability: <b>{prob:.4f}</b><br>
                    Risk Tier: <b>{tier}</b>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.json({c: str(row[c]) for c in display_cols if c in row.index})
            else:
                st.warning(f"Index must be between 0 and {len(df)-1}")
        except ValueError:
            st.error("Please enter a valid integer.")

# ═══════════════════════════════════════════════════════
# PAGE 3 — RISK ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "⚠️ Risk Analysis":
    st.title("⚠️ Risk Analysis — Critical Patterns")
    st.markdown("---")

    critical = df[df["risk_tier"] == "Critical Risk"]
    st.markdown(f"**Critical Risk transactions: {len(critical):,}**")

    # Amount distribution
    if "TransactionAmt_est" in df.columns:
        fig = px.histogram(
            df, x="TransactionAmt_est", color="risk_tier",
            color_discrete_map=TIER_COLORS, log_x=True,
            barmode="overlay", opacity=0.7,
            title="Transaction Amount Distribution by Risk Tier (Log Scale)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Hour of day
    if "hour" in df.columns:
        col1, col2 = st.columns(2)
        hourly_c = critical.groupby("hour").size().reset_index(name="count")
        fig2 = px.bar(
            hourly_c, x="hour", y="count",
            color_discrete_sequence=["#e74c3c"],
            title="Critical Risk — Hour of Day"
        )
        col1.plotly_chart(fig2, use_container_width=True)

        # Night vs day for all tiers
        if "is_night" in df.columns:
            night_by_tier = (df.groupby("risk_tier")["is_night"]
                             .mean() * 100).reset_index()
            night_by_tier.columns = ["Risk Tier", "Night Transaction %"]
            fig3 = px.bar(
                night_by_tier, x="Risk Tier", y="Night Transaction %",
                color="Risk Tier", color_discrete_map=TIER_COLORS,
                title="Night-Time Transaction % by Tier"
            )
            col2.plotly_chart(fig3, use_container_width=True)

    # Round amounts
    if "amt_is_round" in df.columns:
        round_by_tier = (df.groupby("risk_tier")["amt_is_round"]
                         .mean() * 100).reset_index()
        round_by_tier.columns = ["Risk Tier", "Round Amount %"]
        fig4 = px.bar(
            round_by_tier, x="Risk Tier", y="Round Amount %",
            color="Risk Tier", color_discrete_map=TIER_COLORS,
            title="Round Dollar Amount % by Risk Tier"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Fraud Prevention Policy Recommendations")
    st.error(
        "**🔴 Policy 1 — Night-Time High-Value Block**\n\n"
        "Flag all transactions > \$500 occurring between 22:00–05:00 "
        "for mandatory manual review before approval."
    )
    st.warning(
        "**🟡 Policy 2 — Round Amount Anomaly Alert**\n\n"
        "Automatically escalate round-dollar transactions (no cents) "
        "from accounts < 30 days old to the fraud operations team."
    )
    st.success(
        "**🟢 Estimated Annual Savings**\n\n"
        "Based on model recall and average fraud transaction value of \$200, "
        "the system is estimated to prevent **\$1M+** in annual fraud losses."
    )