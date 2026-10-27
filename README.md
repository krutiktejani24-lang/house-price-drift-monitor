
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## How it works

### Data drift
Uses the **Population Stability Index (PSI)** with bins derived from the reference distribution for numeric features. Categorical features use a PSI-style distribution comparison. The default threshold is 0.10; adjust it in the sidebar. Common rules of thumb (0.10 / 0.25) are heuristics, not universal guarantees.


## Project structure

```text
house_price_drift_monitor/
├── app.py                 # Streamlit dashboard and visualizations
├── monitor.py              # Drift calculations and report generation
├── requirements.txt
├── README.md
├── data/
│   └── README.md
└── reports/
    └── .gitkeep
```

## Important note
A true production monitor should schedule this analysis, persist metrics over time, monitor data quality, and alert through your logging/notification stack. This project is a local assignment-ready dashboard and does not automatically connect to a live production endpoint.
