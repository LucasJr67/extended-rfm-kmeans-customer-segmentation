# Extended RFM Customer Segmentation

A reproducible research project for testing whether documented behavioral features add useful structure to traditional RFM customer segmentation. The Streamlit app accepts a user-supplied CSV or XLSX dataset, exposes every source-column mapping, and reports computed results without claiming that clustering metrics demonstrate marketing effectiveness.

## Research question and feature sets

- **Traditional RFM:** Recency (R), Frequency (F), and Monetary value (M).
- **Extended RFM (design):** R, F, M, unique product categories (D), promotional spending share (PS), and purchase-interval variability (PR).
- The repository had no prior pipeline, notebooks, data, result files, or feature definitions to preserve. These six Extended RFM features are an explicit configurable design, not a claim about historical results. No additional features are invented.
- D is available only when a category column is mapped. PS is available only when the user maps a column that genuinely represents promotional-attributed spending; a discount amount is not automatically treated as promotional spending. PR uses standard deviation of the intervals between distinct purchase dates; fewer than three distinct dates yields a missing value, not an assumed zero.
- M is the sum of the mapped transaction-level spend column (which should already represent the amount to attribute to each transaction). Refunds/negative amounts remain in the sum; customers with non-positive totals cannot be used with `log1p` and are excluded with a reported reason.

The feature names, formulas, and default feature lists are centralized in [`src/config.py`](src/config.py). Available features and chosen lists are shown in the app and can be changed there or by passing a different list to the pipeline.

## Requirements and setup (Windows PowerShell)

Use Python 3.13. In PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
streamlit run app\streamlit_app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Plotly PNG export requires Kaleido and a compatible Chrome/Chromium installation; HTML charts and interactive dashboard charts do not require static image export.

## Dataset workflow

1. Start the app and upload a CSV or XLSX file. No sample or generated dataset is used as research evidence.
2. Map customer ID, order/invoice ID, purchase date, and transaction spend. Map product category to enable D. Map promotional spending only if its meaning and units are valid for the PS formula.
3. Review the data preview, exploration, and sequential cleaning report.
4. Choose an analysis date (default: one day after the latest valid purchase date) and inspect the resulting customer features and exclusions.
5. Select the Extended RFM feature list, experiment settings, and explicitly run the experiment. Both feature sets use the same eligible customers, `log1p`, StandardScaler, K range, random seed, and evaluation procedure.
6. Review all metrics, stability, and cluster profiles. Download the metric, profile, and customer-assignment tables and charts. Run-specific outputs are written under `results/tables/` and `results/figures/`; new runs do not replace outputs from different data/settings.

The expected source fields are transaction-level: one or more rows per order, with a stable customer and invoice identifier, a parseable purchase date, and a numeric transaction spend amount. XLSX loading reads the first worksheet. CSV encoding and malformed data errors are shown rather than silently replaced.

## Experimental protocol and interpretation

- The default K range is 2–10, random seed is 42, and K-Means uses `n_init=10`. Settings are configurable in the app and reusable API.
- Each feature is transformed with `log1p` and then standardized. Feature values must be non-negative. The application reports customers omitted because a selected feature is missing or the spending total is non-positive. It does not substitute invented values.
- Metrics: Silhouette (higher), Davies–Bouldin (lower), Calinski–Harabasz (higher), and Dunn (higher). Dunn is the minimum inter-cluster Euclidean distance divided by the maximum within-cluster Euclidean diameter. For datasets larger than the configured evaluation sample limit, all four metrics use one reproducibly, cluster-stratified sample; the dashboard identifies that sample size.
- Stability is the mean and population standard deviation of pairwise Adjusted Rand Index (ARI) across all pairs of repeated K-Means fits at the same K, calculated on all eligible customers. Seeds are `random_state + repeat_index`.
- The declared per-feature-set K selection rule is maximum Silhouette, with the smaller K breaking ties. All metrics are still shown; a best K is not produced when Silhouette is undefined. Metric values from different feature spaces are not proof that one segmentation is objectively superior; notably, Calinski–Harabasz values should not be compared alone across spaces.
- Marketing notes are deterministic rules based on a cluster's median feature values relative to the dataset median. They are descriptive prompts for human review, not causal findings, targeting recommendations, or evidence of campaign effectiveness.

## Project layout

```text
src/
  ├── config.py              Feature definitions, business constants, and project-wide defaults
  ├── preprocessing.py       Input loading, exploration, and transaction cleaning
  ├── features.py            Customer-level feature engineering (R, F, M, D, PS, PR)
  ├── evaluation.py          Clustering geometric metrics & stratified sampling
  ├── clustering.py          K-Means experiments, ARI stability, and Best K selection
  ├── visualization.py       Plotly figures (metrics by K, PCA 2D) and cluster profiling
  ├── business_validation.py CLV projection, retention rates, ANOVA separation, campaign simulation
  └── pipeline_manager.py    Model serialization (.joblib) and new customer batch scoring
app/
  └── streamlit_app.py       3-Tab Interactive dashboard (Train, Business Validation, Batch Inference)
tests/                       29 unit and integration tests covering all modules
results/
  ├── tables/                Generated experiment and profile CSVs
  ├── figures/               Generated chart files (HTML & PNG)
  └── models/                Serialized model pipelines (.joblib)
data/raw/                    Private source datasets (ignored by Git)
```

The app works directly with an uploaded file, so placing data in `data/raw/` is optional. Never commit identifiable or otherwise sensitive customer data.

## Reusable API

### 1. Training & Clustering Experiment

```python
from datetime import date
from src.config import ColumnMapping
from src.preprocessing import clean_transactions, load_dataset
from src.features import build_customer_features
from src.clustering import run_experiments

raw = load_dataset("data/raw/your_data.csv")
mapping = ColumnMapping(
    customer_id="CustomerID",
    order_id="InvoiceNo",
    purchase_date="InvoiceDate",
    spend="TransactionSpend",
    category="ProductCategory",       # optional
    promotional_spend="PromoSpend",   # optional, only with a valid source definition
)
transactions, cleaning_report = clean_transactions(raw, mapping)
customers, feature_report = build_customer_features(transactions, mapping, date.today())
experiment = run_experiments(
    customers,
    {"Traditional RFM": ("R", "F", "M"), "Extended RFM": ("R", "F", "M", "D", "PR")},
)
print(experiment.results)
```

### 2. Business Validation & Economic Metrics

```python
from src.business_validation import (
    calculate_customer_clv,
    summarize_cluster_clv,
    calculate_retention_metrics,
    simulate_promotional_campaign,
)

labels = experiment.labels[("Extended RFM", 2)]
# Projected Customer Lifetime Value
clv = calculate_customer_clv(experiment.eligible_customer_features, profit_margin=0.20)
clv_summary = summarize_cluster_clv(experiment.eligible_customer_features, labels, "Extended RFM", 2)

# Retention & Repurchase Rates
retention = calculate_retention_metrics(experiment.eligible_customer_features, labels, "Extended RFM", 2)

# Marketing Campaign Simulation (Targeting Cluster 0)
campaign = simulate_promotional_campaign(
    experiment.eligible_customer_features,
    labels,
    target_cluster=0,
    discount_rate=0.10,
    expected_response_rate=0.15,
)
print(f"Projected Campaign ROI: {campaign['roi_pct']:.1f}%")
```

### 3. Model Persistence & Batch Customer Inference

```python
from src.pipeline_manager import (
    create_pipeline_from_experiment,
    save_pipeline,
    load_pipeline,
    predict_customer_segments,
)

# Save fitted pipeline
pipeline = create_pipeline_from_experiment(
    eligible_customer_features=experiment.eligible_customer_features,
    features=("R", "F", "M", "D", "PR"),
    model_name="Extended RFM",
    k=2,
    random_state=42,
    n_init=10,
    cluster_profiles=profile_clusters(experiment.eligible_customer_features, labels, "Extended RFM", 2),
    run_id="run_01",
)
save_pipeline(pipeline, "results/models/extended_rfm_k2.joblib")

# Load and predict on new customers
loaded = load_pipeline("results/models/extended_rfm_k2.joblib")
new_assignments = predict_customer_segments(loaded, new_customer_features)
print(new_assignments[["predicted_cluster", "segment_description"]])
```

## Limitations

Results depend on source-data semantics, the selected features, scaling, and K-Means assumptions. Customers with insufficient purchase history have no PR value and are excluded when PR is selected so both models retain a comparable population. A missing category or valid promotional-spend column prevents that feature from being computed. Silhouette and other internal indices assess geometric properties only, which is why the dashboard complements them with economic validation (CLV, Retention, and Campaign ROI simulation).

