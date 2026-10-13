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
src/                 Reusable loading, cleaning, feature, clustering, and evaluation code
app/streamlit_app.py Interactive research dashboard
tests/               Formula, edge-case, metric, and end-to-end tests
data/raw/            Put private source datasets here (ignored by Git)
data/processed/      Optional local processed data (ignored by Git)
results/tables/      Generated experiment and profile CSVs (ignored by Git)
results/figures/     Generated chart files (ignored by Git)
```

The app works directly with an uploaded file, so placing data in `data/raw/` is optional. Never commit identifiable or otherwise sensitive customer data.

## Reusable API

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

The example mapping is illustrative: replace every column name with a verified field from your own dataset. `PS` is intentionally omitted from the example feature set because its source semantics cannot be presumed.

## Limitations

Results depend on source-data semantics, the selected features, scaling, and K-Means assumptions. Customers with insufficient purchase history have no PR value and are excluded when PR is selected so both models retain a comparable population. A missing category or valid promotional-spend column prevents that feature from being computed. Silhouette and other internal indices assess geometric properties only; they do not measure business value, customer lifetime value, or campaign lift.
