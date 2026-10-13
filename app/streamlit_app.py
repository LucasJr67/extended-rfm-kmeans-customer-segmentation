"""Interactive dashboard for the Extended RFM research workflow."""

from datetime import date
import hashlib
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

from src.clustering import run_experiments, select_best_k
from src.config import (
    DEFAULT_K_VALUES,
    DEFAULT_METRIC_SAMPLE_LIMIT,
    DEFAULT_N_INIT,
    DEFAULT_RANDOM_STATE,
    FEATURE_DEFINITIONS,
    FEATURE_SETS,
    ColumnMapping,
)
from src.features import available_features, build_customer_features
from src.preprocessing import clean_transactions, explore_dataset
from src.visualization import cluster_scatter_figure, metric_comparison_figure, profile_clusters


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"


@st.cache_data(show_spinner=False)
def read_uploaded_file(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(content))
    if suffix == ".xlsx":
        return pd.read_excel(BytesIO(content))
    raise ValueError("Please upload a CSV or XLSX file.")


def _column_selector(label: str, columns: list[str], required: bool) -> str | None:
    options = columns if required else ["(not mapped)"] + columns
    selected = st.selectbox(label, options=options, key=f"mapping_{label}")
    return selected if required or selected != "(not mapped)" else None


def main() -> None:
    st.set_page_config(page_title="Extended RFM Research", layout="wide")
    st.title("Extended RFM Customer Segmentation")
    st.caption(
        "Research objective: test whether mapped behavioral features add useful "
        "geometric structure to traditional RFM. Internal metrics do not establish "
        "marketing effectiveness."
    )
    with st.expander("Research definitions and assumptions", expanded=True):
        st.markdown(
            "- Traditional RFM = R, F, M. Configured Extended RFM = R, F, M, D, PS, PR.\n"
            "- F counts unique invoices, not transaction rows. M sums the mapped spend field.\n"
            "- PS requires a verified promotional-spending amount, not an assumed discount.\n"
            "- PR is missing for customers with fewer than three distinct purchase dates.\n"
            "- `log1p` and StandardScaler are applied consistently; selected-feature gaps "
            "exclude the same customers from every model."
        )
    uploaded = st.file_uploader("Upload transaction data", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Upload a CSV or Excel workbook to begin. No sample results are shown.")
        return

    try:
        raw = read_uploaded_file(uploaded.getvalue(), uploaded.name)
    except (ValueError, OSError, pd.errors.ParserError, ImportError) as error:
        st.error(f"Could not read the uploaded dataset: {error}")
        return
    if raw.empty:
        st.error("The uploaded dataset contains no rows.")
        return
    raw.columns = raw.columns.map(str)
    st.subheader("Dataset preview and source-column mapping")
    st.write(f"{len(raw):,} transaction rows × {len(raw.columns):,} columns")
    st.dataframe(raw.head(20), use_container_width=True)

    columns = [str(column) for column in raw.columns]
    left, right = st.columns(2)
    with left:
        customer_id = _column_selector("Customer ID (required)", columns, True)
        order_id = _column_selector("Order / invoice ID (required)", columns, True)
        purchase_date = _column_selector("Purchase date (required)", columns, True)
    with right:
        spend = _column_selector("Transaction spend (required)", columns, True)
        category = _column_selector("Product category (for D)", columns, False)
        promotional_spend = _column_selector(
            "Promotional-attributed spend (for PS; verified meaning only)",
            columns,
            False,
        )
    mapping = ColumnMapping(
        customer_id=customer_id,
        order_id=order_id,
        purchase_date=purchase_date,
        spend=spend,
        category=category,
        promotional_spend=promotional_spend,
    )

    st.subheader("Data quality and preprocessing")
    try:
        exploration = explore_dataset(raw)
        cleaned, cleaning_report = clean_transactions(raw, mapping)
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Dataset validation failed: {error}")
        return
    quality_left, quality_right = st.columns(2)
    with quality_left:
        st.write("Column types")
        st.dataframe(
            pd.DataFrame(
                {"column": exploration["columns"], "dtype": list(exploration["dtypes"].values())}
            ),
            hide_index=True,
            use_container_width=True,
        )
    with quality_right:
        st.write("Missing values and exact duplicates before cleaning")
        st.dataframe(
            pd.DataFrame(
                {
                    "column": list(exploration["missing_values"]),
                    "missing": list(exploration["missing_values"].values()),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.metric("Exact duplicate rows", exploration["duplicate_rows"])
    st.write("Descriptive statistics")
    st.dataframe(exploration["describe"], use_container_width=True)
    st.write("Sequential cleaning report")
    st.dataframe(
        pd.DataFrame(
            [{"cleaning_step": name, "row_count": count} for name, count in cleaning_report.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Customer features")
    try:
        parsed_max = pd.to_datetime(cleaned[mapping.purchase_date]).max().date()
        default_analysis_date = date.fromordinal(parsed_max.toordinal() + 1)
        analysis_date = st.date_input("Analysis date", value=default_analysis_date)
        customer_features, feature_report = build_customer_features(
            cleaned, mapping, analysis_date
        )
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        st.error(f"Could not engineer customer features: {error}")
        return
    st.caption(
        f"Analysis date: {feature_report['analysis_date']} · "
        f"{feature_report['customer_count']:,} customers before experiment exclusions"
    )
    st.dataframe(customer_features.head(30), use_container_width=True)
    st.write("Available formulas")
    st.dataframe(
        pd.DataFrame(
            [
                {"feature": name, "definition": FEATURE_DEFINITIONS[name]}
                for name in feature_report["available_features"]
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.write("Missing feature values")
    st.json(feature_report["missing_feature_values"])
    if mapping.promotional_spend is None:
        st.info("PS is unavailable: map a source field only if it truly represents promotional spending.")
    if mapping.category is None:
        st.info("D is unavailable because no product category field is mapped.")
    if feature_report["insufficient_purchase_history_for_PR"]:
        st.warning(
            f"PR is unavailable for {feature_report['insufficient_purchase_history_for_PR']:,} "
            "customers with fewer than three distinct purchase dates."
        )

    st.subheader("Run reproducible clustering experiments")
    available = available_features(mapping)
    default_extended = [f for f in FEATURE_SETS["extended_rfm"] if f in available]
    selected_extended = st.multiselect(
        "Extended model features (select only available, semantically valid features)",
        options=list(available),
        default=default_extended,
    )
    k_values = st.multiselect("K values", options=list(range(2, 11)), default=list(DEFAULT_K_VALUES))
    settings_left, settings_right = st.columns(2)
    with settings_left:
        random_state = st.number_input("Random seed", min_value=0, value=DEFAULT_RANDOM_STATE)
        n_init = st.number_input("K-Means n_init", min_value=1, value=DEFAULT_N_INIT)
    with settings_right:
        stability_runs = st.number_input("Repeated fits for ARI stability", min_value=2, value=5)
        sample_limit = st.number_input(
            "Maximum rows used for each internal metric",
            min_value=2,
            value=DEFAULT_METRIC_SAMPLE_LIMIT,
        )
    signature_payload = repr(
        (
            hashlib.sha256(uploaded.getvalue()).hexdigest(),
            uploaded.name,
            mapping,
            analysis_date.isoformat(),
            tuple(selected_extended),
            tuple(k_values),
            int(random_state),
            int(n_init),
            int(stability_runs),
            int(sample_limit),
        )
    )
    experiment_signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    if "experiment_result" not in st.session_state:
        st.session_state.experiment_result = None
    if st.button("Run clustering experiments", type="primary"):
        if len(selected_extended) < 2:
            st.error("Select at least two available features for the Extended model.")
            return
        if not k_values:
            st.error("Select at least one K value.")
            return
        models = {
            "Traditional RFM": FEATURE_SETS["traditional_rfm"],
            "Extended RFM": tuple(selected_extended),
        }
        if models["Extended RFM"] == models["Traditional RFM"]:
            st.error("The Extended model must use a feature list different from Traditional RFM.")
            return
        with st.spinner("Fitting K-Means models and calculating evaluation metrics..."):
            try:
                result = run_experiments(
                    customer_features,
                    models,
                    k_values=k_values,
                    random_state=int(random_state),
                    n_init=int(n_init),
                    stability_runs=int(stability_runs),
                    metric_sample_limit=int(sample_limit),
                )
            except (ValueError, KeyError, FloatingPointError) as error:
                st.error(f"Experiment could not run: {error}")
                return
        st.session_state.experiment_result = result
        st.session_state.experiment_models = models
        st.session_state.experiment_analysis_date = feature_report["analysis_date"]
        st.session_state.experiment_signature = experiment_signature
        st.session_state.pop("metric_chart_png", None)
        st.session_state.pop("metric_chart_filename", None)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz", "dunn"):
            metric_comparison_figure(result.results, metric).write_html(
                FIGURES_DIR
                / f"{metric}_by_k_{experiment_signature[:12]}.html"
            )

    result = st.session_state.experiment_result
    if result is None or st.session_state.get("experiment_signature") != experiment_signature:
        st.info("Run experiments for the current dataset, feature selection, and settings.")
        return
    st.subheader("Model comparison")
    st.caption(
        f"Same eligible population: {result.exclusions['eligible_customers']:,} customers; "
        f"{result.metric_sample_size:,} rows per metric calculation "
        f"(limit {result.metric_sample_limit:,}). Stability is computed on all eligible customers."
    )
    st.write("Population exclusions")
    st.json(result.exclusions)
    st.dataframe(result.results, hide_index=True, use_container_width=True)
    best = select_best_k(result.results)
    if best.empty:
        st.warning("No best K selected: Silhouette was undefined for every tested K.")
    else:
        st.write("Declared selection: maximum Silhouette within each feature set; ties choose smaller K.")
        st.dataframe(best, hide_index=True, use_container_width=True)

    st.write("Internal metric comparison")
    st.caption(
        "Internal metrics describe geometry, not business value. Metric scores depend on "
        "feature space; do not use Calinski–Harabasz alone to rank different feature sets."
    )
    metric_names = ("silhouette", "davies_bouldin", "calinski_harabasz", "dunn")
    for metric in metric_names:
        st.plotly_chart(metric_comparison_figure(result.results, metric), use_container_width=True)
    selected_metric = st.selectbox("Metric chart to export as PNG", metric_names)
    metric_figure = metric_comparison_figure(result.results, selected_metric)
    metric_filename = f"{selected_metric}_by_k_{experiment_signature[:12]}.png"
    if st.button("Prepare metric chart PNG"):
        try:
            metric_image = metric_figure.to_image(format="png")
        except (ImportError, RuntimeError, ValueError, OSError) as error:
            st.error(f"PNG export failed. Install Kaleido and Chrome/Chromium: {error}")
        else:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            (FIGURES_DIR / metric_filename).write_bytes(metric_image)
            st.session_state.metric_chart_png = metric_image
            st.session_state.metric_chart_filename = metric_filename
    if st.session_state.get("metric_chart_filename") == metric_filename:
        st.download_button(
            "Download selected metric chart (PNG)",
            data=st.session_state.metric_chart_png,
            file_name=metric_filename,
            mime="image/png",
        )

    st.subheader("Cluster profiles")
    choices = [
        (name, k)
        for name, k in result.labels
    ]
    if choices:
        labels_display = [f"{name} · K={k}" for name, k in choices]
        selected_display = st.selectbox("Profile and visualize a fitted model", labels_display)
        choice = choices[labels_display.index(selected_display)]
        profile = profile_clusters(
            result.eligible_customer_features,
            result.labels[choice],
            choice[0],
            choice[1],
        )
        st.dataframe(profile, hide_index=True, use_container_width=True)
        st.caption(
            "Observations compare cluster medians with overall medians. They are not validated "
            "marketing recommendations or causal conclusions."
        )
        st.plotly_chart(
            cluster_scatter_figure(
                result.scaled_features[choice[0]],
                result.labels[choice],
                f"{choice[0]} PCA projection · K={choice[1]}",
            ),
            use_container_width=True,
        )
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        result_name = f"{choice[0].lower().replace(' ', '_')}_k{choice[1]}"
        scatter = cluster_scatter_figure(
            result.scaled_features[choice[0]],
            result.labels[choice],
            f"{choice[0]} PCA projection · K={choice[1]}",
        )
        try:
            image = scatter.to_image(format="png")
        except (ImportError, RuntimeError, ValueError, OSError) as error:
            st.error(f"PNG export failed. Install Kaleido and Chrome/Chromium: {error}")
        else:
            st.download_button(
                "Download cluster chart (PNG)",
                data=image,
                file_name=f"{result_name}_pca.png",
                mime="image/png",
            )
            (FIGURES_DIR / f"{result_name}_pca.png").write_bytes(image)
        scatter.write_html(FIGURES_DIR / f"{result_name}_pca.html")
    else:
        profile = pd.DataFrame()

    st.subheader("Download and saved outputs")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    run_id = experiment_signature[:12]
    results_csv = result.results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metric results (CSV)",
        data=results_csv,
        file_name="clustering_metrics.csv",
        mime="text/csv",
    )
    all_profiles = []
    for (model, k), labels in result.labels.items():
        all_profiles.append(
            profile_clusters(result.eligible_customer_features, labels, model, k)
        )
    profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    profiles_csv = profiles.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cluster profiles (CSV)",
        data=profiles_csv,
        file_name="cluster_profiles.csv",
        mime="text/csv",
    )
    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "customer_id": labels.index.astype(str),
                    "model": model,
                    "k": k,
                    "cluster": labels.to_numpy(),
                }
            )
            for (model, k), labels in result.labels.items()
        ],
        ignore_index=True,
    ) if result.labels else pd.DataFrame(
        columns=["customer_id", "model", "k", "cluster"]
    )
    assignments_csv = assignments.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download customer assignments (CSV)",
        data=assignments_csv,
        file_name="customer_assignments.csv",
        mime="text/csv",
    )
    result.results.to_csv(TABLES_DIR / f"clustering_metrics_{run_id}.csv", index=False)
    profiles.to_csv(TABLES_DIR / f"cluster_profiles_{run_id}.csv", index=False)
    assignments.to_csv(TABLES_DIR / f"customer_assignments_{run_id}.csv", index=False)
    st.caption("CSV tables and interactive HTML/PCA chart files are saved under results/.")


if __name__ == "__main__":
    main()
