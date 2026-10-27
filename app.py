from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from monitor import run_monitoring, load_sample_data

ROOT = Path(__file__).parent
st.set_page_config(page_title="House Price Model Monitor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction — Drift Monitoring")
st.caption("Data drift, concept drift, and monitoring report")

with st.sidebar:
    st.header("Monitoring settings")
    threshold = st.slider("Data drift threshold (%)", 1, 50, 10)
    performance_drop = st.slider("Performance degradation threshold (%)", 1, 100, 20)
    st.markdown("---")
    st.write("**Reference dataset:** historical / training data")
    st.write("**Current dataset:** recent production data with actual prices")

st.markdown("""
This demo compares a **reference dataset** with recent production observations.
Data drift checks changes in input features; concept drift checks whether prediction
errors have worsened after actual sale prices become available.
""")

uploaded_ref = st.file_uploader("Upload reference CSV (optional)", type=["csv"], key="ref")
uploaded_cur = st.file_uploader("Upload current production CSV (optional)", type=["csv"], key="cur")

if uploaded_ref is not None and uploaded_cur is not None:
    reference = pd.read_csv(uploaded_ref)
    current = pd.read_csv(uploaded_cur)
    data_source = "Uploaded CSV files"
else:
    reference, current = load_sample_data()
    data_source = "Built-in synthetic sample data"
    st.info("Using built-in synthetic sample data. Upload both CSV files to monitor your own data.")

with st.expander("Expected CSV columns and sample data"):
    st.code("area_sqft,bedrooms,location,predicted_price,actual_price,date", language="text")
    st.write("Reference data should contain input features. Current data should contain the same features and, for concept drift, `predicted_price` and `actual_price`.")
    st.dataframe(current.head(8), use_container_width=True)

if st.button("Run monitoring", type="primary"):
    result = run_monitoring(reference, current, threshold / 100, performance_drop / 100)
    st.session_state["monitor_result"] = result
    st.session_state["monitor_source"] = data_source

if "monitor_result" in st.session_state:
    result = st.session_state["monitor_result"]
    st.caption(f"Data source: {st.session_state.get('monitor_source', data_source)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current records", f"{len(current):,}")
    c2.metric("Features checked", len(result["drift"]))
    c3.metric("Features drifted", sum(x["drift_detected"] for x in result["drift"].values()))
    c4.metric("Concept drift", "Detected" if result["concept"]["drift_detected"] else ("Not detected" if result["concept"]["available"] else "N/A"))

    st.subheader("1. Data drift by feature")
    drift_df = pd.DataFrame([
        {"Feature": name, "Type": v["type"], "Reference": v["reference_summary"],
         "Current": v["current_summary"], "Drift score": v["score"],
         "Drift detected": "Yes" if v["drift_detected"] else "No"}
        for name, v in result["drift"].items()
    ])
    st.dataframe(drift_df, use_container_width=True, hide_index=True)

    numeric_features = [k for k, v in result["drift"].items() if v["type"] == "numeric"]
    if numeric_features:
        st.markdown("**Numeric feature distributions**")
        for feature in numeric_features:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(pd.to_numeric(reference[feature], errors="coerce").dropna(), bins=25, alpha=0.55, density=True, label="Reference")
            ax.hist(pd.to_numeric(current[feature], errors="coerce").dropna(), bins=25, alpha=0.55, density=True, label="Current")
            ax.set_title(f"{feature}: reference vs current")
            ax.set_ylabel("Density")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    categorical_features = [k for k, v in result["drift"].items() if v["type"] == "categorical"]
    if categorical_features:
        st.markdown("**Categorical feature distribution changes**")
        for feature in categorical_features:
            combined = pd.concat([
                reference[feature].astype(str).value_counts(normalize=True).rename("Reference"),
                current[feature].astype(str).value_counts(normalize=True).rename("Current")
            ], axis=1).fillna(0)
            fig, ax = plt.subplots(figsize=(8, 3))
            combined.plot(kind="bar", ax=ax)
            ax.set_title(f"{feature}: category proportions")
            ax.set_ylabel("Proportion")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("2. Concept drift / model performance")
    concept = result["concept"]
    if concept["available"]:
        m1, m2, m3 = st.columns(3)
        m1.metric("Reference MAE", f"{concept['reference_mae']:,.0f}")
        m2.metric("Current MAE", f"{concept['current_mae']:,.0f}")
        m3.metric("MAE change", f"{concept['mae_change_pct']:+.1f}%")
        st.write(concept["message"])
        if "date" in current.columns:
            timeline = current.copy()
            timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
            timeline["absolute_error"] = (pd.to_numeric(timeline["actual_price"], errors="coerce") - pd.to_numeric(timeline["predicted_price"], errors="coerce")).abs()
            timeline = timeline.dropna(subset=["date", "absolute_error"]).sort_values("date")
            if not timeline.empty:
                timeline["period"] = timeline["date"].dt.to_period("W").astype(str)
                weekly = timeline.groupby("period", as_index=False)["absolute_error"].mean()
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(weekly["period"], weekly["absolute_error"], marker="o")
                ax.set_title("Weekly mean absolute prediction error")
                ax.set_ylabel("MAE (price units)")
                ax.tick_params(axis="x", rotation=35)
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.warning("Concept drift could not be evaluated: current data needs `predicted_price` and `actual_price`. To compare reference performance too, add these columns to the reference CSV.")

    st.subheader("3. Monitoring summary")
    for item in result["recommendations"]:
        st.write(f"- {item}")

    report = result["report_markdown"]
    st.download_button("Download monitoring report (Markdown)", report, file_name="house_price_monitoring_report.md", mime="text/markdown")
    st.download_button("Download feature drift results (CSV)", drift_df.to_csv(index=False), file_name="feature_drift_results.csv", mime="text/csv")
