from datetime import datetime
import numpy as np
import pandas as pd

NUMERIC_CANDIDATES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
TARGET_COLUMNS = {"predicted_price", "actual_price", "date"}

def load_sample_data(seed=42):
    """Create reproducible synthetic reference and production datasets."""
    rng = np.random.default_rng(seed)
    n_ref, n_cur = 800, 300
    ref_area = np.clip(rng.normal(1500, 450, n_ref), 450, 4200)
    ref_bedrooms = rng.choice([1, 2, 3, 4, 5], n_ref, p=[.08, .28, .38, .20, .06])
    ref_locations = rng.choice(["Central", "Suburb", "Rural", "Waterfront"], n_ref, p=[.28, .42, .22, .08])
    location_effect = {"Central": 1.25, "Suburb": 1.0, "Rural": .68, "Waterfront": 1.5}
    ref_price = np.array([area * 180 * location_effect[loc] + beds * 90000 for area, beds, loc in zip(ref_area, ref_bedrooms, ref_locations)])
    ref_price += rng.normal(0, 45000, n_ref)
    ref_pred = ref_price + rng.normal(0, 65000, n_ref)
    reference = pd.DataFrame({
        "area_sqft": ref_area.round(0).astype(int),
        "bedrooms": ref_bedrooms,
        "location": ref_locations,
        "predicted_price": ref_pred.round(0),
        "actual_price": ref_price.round(0),
    })

    # Production population shifts toward larger homes and more expensive locations.
    cur_area = np.clip(rng.normal(1950, 520, n_cur), 450, 5000)
    cur_bedrooms = rng.choice([1, 2, 3, 4, 5], n_cur, p=[.03, .17, .34, .30, .16])
    cur_locations = rng.choice(["Central", "Suburb", "Rural", "Waterfront", "New Development"], n_cur, p=[.20, .32, .16, .20, .12])
    current_effect = {"Central": 1.35, "Suburb": 1.08, "Rural": .72, "Waterfront": 1.72, "New Development": 1.18}
    cur_price = np.array([area * 205 * current_effect[loc] + beds * 105000 for area, beds, loc in zip(cur_area, cur_bedrooms, cur_locations)])
    # Larger error to simulate a model that is becoming stale.
    cur_pred = cur_price + rng.normal(-45000, 145000, n_cur)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_cur, freq="D")
    current = pd.DataFrame({
        "area_sqft": cur_area.round(0).astype(int),
        "bedrooms": cur_bedrooms,
        "location": cur_locations,
        "predicted_price": cur_pred.round(0),
        "actual_price": cur_price.round(0),
        "date": dates.astype(str),
    })
    reference["date"] = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n_ref, freq="D").astype(str)
    return reference, current

def _numeric_drift(ref, cur):
    """Population Stability Index (PSI) with reference-derived bins."""
    ref = pd.to_numeric(ref, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(cur, errors="coerce").dropna().to_numpy()
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    psi = np.sum((c - r) * np.log((c + eps) / (r + eps)))
    return float(psi)

def _categorical_drift(ref, cur):
    ref = ref.fillna("<MISSING>").astype(str)
    cur = cur.fillna("<MISSING>").astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if not categories:
        return 0.0
    r = ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    eps = 1e-6
    return float(np.sum((c - r) * np.log((c + eps) / (r + eps))))

def _summary(series, kind):
    if kind == "numeric":
        x = pd.to_numeric(series, errors="coerce").dropna()
        return f"mean={x.mean():,.2f}, std={x.std():,.2f}"
    return f"{series.astype(str).nunique()} categories"

def run_monitoring(reference, current, drift_threshold=0.10, performance_drop_threshold=0.20):
    if reference.empty or current.empty:
        raise ValueError("Reference and current datasets must both contain at least one row.")
    features = [c for c in reference.columns if c in current.columns and c not in TARGET_COLUMNS]
    if not features:
        raise ValueError("No shared input features found. Expected columns such as area_sqft, bedrooms, location.")
    drift = {}
    for feature in features:
        ref, cur = reference[feature], current[feature]
        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
            kind = "numeric"
            score = _numeric_drift(ref, cur)
            detected = score >= drift_threshold
        else:
            kind = "categorical"
            score = _categorical_drift(ref, cur)
            detected = score >= drift_threshold
        drift[feature] = {
            "type": kind,
            "score": round(score, 4),
            "drift_detected": bool(detected),
            "reference_summary": _summary(ref, kind),
            "current_summary": _summary(cur, kind),
        }

    concept = {"available": False, "drift_detected": False, "message": "Concept drift unavailable."}
    required = {"predicted_price", "actual_price"}
    if required.issubset(reference.columns) and required.issubset(current.columns):
        ref_actual = pd.to_numeric(reference["actual_price"], errors="coerce")
        ref_pred = pd.to_numeric(reference["predicted_price"], errors="coerce")
        cur_actual = pd.to_numeric(current["actual_price"], errors="coerce")
        cur_pred = pd.to_numeric(current["predicted_price"], errors="coerce")
        ref_err = (ref_actual - ref_pred).abs().dropna()
        cur_err = (cur_actual - cur_pred).abs().dropna()
        if not ref_err.empty and not cur_err.empty:
            ref_mae, cur_mae = float(ref_err.mean()), float(cur_err.mean())
            change = ((cur_mae - ref_mae) / max(ref_mae, 1e-9)) * 100
            detected = change / 100 >= performance_drop_threshold
            concept = {
                "available": True,
                "reference_mae": ref_mae,
                "current_mae": cur_mae,
                "mae_change_pct": change,
                "drift_detected": bool(detected),
                "message": ("⚠️ Performance degradation detected. Investigate market changes and consider retraining."
                            if detected else "No performance degradation beyond the configured threshold."),
            }

    drifted = [k for k, v in drift.items() if v["drift_detected"]]
    recommendations = []
    recommendations.append(f"Data drift: {len(drifted)} of {len(drift)} shared input features crossed the configured PSI threshold ({drift_threshold:.2f}).")
    if drifted:
        recommendations.append("Review shifted features: " + ", ".join(drifted) + ". Validate data pipelines and check whether retraining is needed.")
    if concept["available"]:
        recommendations.append(f"Concept drift: current MAE is {concept['mae_change_pct']:+.1f}% versus reference.")
        if concept["drift_detected"]:
            recommendations.append("Investigate price-market changes, segment errors by location/home size, and retrain only after validating fresh labels.")
    else:
        recommendations.append("Concept drift was not evaluated because actual/predicted price columns are missing in one or both datasets.")
    recommendations.append("This assignment uses synthetic data by default; replace it with production/reference CSVs before making operational decisions.")

    report_lines = [
        "# House Price Prediction — Monitoring Report",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"Reference rows: {len(reference)} | Current rows: {len(current)}",
        "",
        "## Data drift",
        f"Configured PSI threshold: {drift_threshold:.2f}",
        "",
        "| Feature | Type | Drift score (PSI) | Drift detected | Reference summary | Current summary |",
        "|---|---|---:|---|---|---|",
    ]
    for name, item in drift.items():
        report_lines.append(f"| {name} | {item['type']} | {item['score']:.4f} | {'Yes' if item['drift_detected'] else 'No'} | {item['reference_summary']} | {item['current_summary']} |")
    report_lines += ["", "## Concept drift / performance", ""]
    if concept["available"]:
        report_lines += [
            f"- Reference MAE: {concept['reference_mae']:,.2f}",
            f"- Current MAE: {concept['current_mae']:,.2f}",
            f"- MAE change: {concept['mae_change_pct']:+.2f}%",
            f"- Concept drift detected: {'Yes' if concept['drift_detected'] else 'No'}",
            f"- Interpretation: {concept['message']}",
        ]
    else:
        report_lines.append("Not evaluated: both datasets need predicted_price and actual_price.")
    report_lines += ["", "## Recommendations", ""] + [f"- {r}" for r in recommendations]
    return {"drift": drift, "concept": concept, "recommendations": recommendations, "report_markdown": "\n".join(report_lines)}
