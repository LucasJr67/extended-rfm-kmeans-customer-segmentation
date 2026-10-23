"""Interactive research dashboard for Extended RFM Customer Segmentation.

Includes:
1. Training, model comparison, and internal geometric validation.
2. Business validation (CLV projection, Retention & Repurchase, Campaign Simulation).
3. Model persistence and batch customer inference.
"""

from datetime import date
import hashlib
from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.business_validation import (
    calculate_customer_clv,
    calculate_retention_metrics,
    compare_business_separation,
    simulate_promotional_campaign,
    summarize_cluster_clv,
)
from src.clustering import run_experiments, select_best_k
from src.config import (
    DEFAULT_K_VALUES,
    DEFAULT_METRIC_SAMPLE_LIMIT,
    DEFAULT_N_INIT,
    DEFAULT_PROFIT_MARGIN,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RETENTION_RECENCY_DAYS,
    DEFAULT_TIME_HORIZON_YEARS,
    FEATURE_DEFINITIONS,
    FEATURE_SETS,
    ColumnMapping,
)
from src.features import available_features, build_customer_features
from src.pipeline_manager import (
    SegmentationPipeline,
    create_pipeline_from_experiment,
    list_saved_pipelines,
    load_pipeline,
    predict_customer_segments,
    save_pipeline,
)
from src.preprocessing import clean_transactions, explore_dataset
from src.visualization import cluster_scatter_figure, metric_comparison_figure, profile_clusters

TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "results" / "models"


@st.cache_data(show_spinner=False)
def read_uploaded_file(content: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or XLSX files from bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(content))
    if suffix == ".xlsx":
        return pd.read_excel(BytesIO(content))
    raise ValueError("Please upload a CSV or XLSX file.")


def _column_selector(label: str, columns: list[str], required: bool, key_prefix: str = "mapping") -> str | None:
    options = columns if required else ["(not mapped)"] + columns
    selected = st.selectbox(label, options=options, key=f"{key_prefix}_{label}")
    return selected if required or selected != "(not mapped)" else None


def _css_gradient(series: pd.Series, low_rgb: tuple, high_rgb: tuple, reverse: bool = False) -> list[str]:
    """Return a list of CSS background-color strings interpolated between two RGB colors."""
    vals = pd.to_numeric(series, errors="coerce")
    vmin, vmax = vals.min(), vals.max()
    styles: list[str] = []
    for v in vals:
        if pd.isna(v) or vmax == vmin:
            styles.append("")
            continue
        ratio = (v - vmin) / (vmax - vmin)
        if reverse:
            ratio = 1.0 - ratio
        r = int(low_rgb[0] + (high_rgb[0] - low_rgb[0]) * ratio)
        g = int(low_rgb[1] + (high_rgb[1] - low_rgb[1]) * ratio)
        b = int(low_rgb[2] + (high_rgb[2] - low_rgb[2]) * ratio)
        styles.append(f"background-color: rgb({r},{g},{b})")
    return styles


def _color_metrics_row(df: pd.DataFrame) -> pd.DataFrame:
    """Apply gradient coloring to metric columns without matplotlib."""
    out = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in ("silhouette", "dunn"):
        if col in df.columns:
            out[col] = _css_gradient(df[col], (255, 255, 255), (187, 247, 208))
    if "davies_bouldin" in df.columns:
        out["davies_bouldin"] = _css_gradient(df["davies_bouldin"], (255, 255, 255), (254, 226, 226), reverse=True)
    if "calinski_harabasz" in df.columns:
        out["calinski_harabasz"] = _css_gradient(df["calinski_harabasz"], (255, 255, 255), (191, 219, 254))
    return out


# ==============================================================================
# TAB 1: TRAINING & GEOMETRIC VALIDATION
# ==============================================================================
def render_training_tab() -> None:
    with st.expander("Research definitions and assumptions", expanded=False):
        st.markdown(
            "- Traditional RFM = R, F, M. Configured Extended RFM = R, F, M, D, PS, PR.\n"
            "- F counts unique invoices, not transaction rows. M sums the mapped spend field.\n"
            "- PS requires a verified promotional-spending amount, not an assumed discount.\n"
            "- PR is missing for customers with fewer than three distinct purchase dates.\n"
            "- `log1p` and StandardScaler are applied consistently; selected-feature gaps "
            "exclude the same customers from every model."
        )

    uploaded = st.file_uploader("Upload transaction data", type=["csv", "xlsx"], key="train_uploader")
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
        st.session_state.random_state = int(random_state)
        st.session_state.n_init = int(n_init)
        st.session_state.pop("metric_chart_png", None)
        st.session_state.pop("metric_chart_filename", None)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        selected_k = select_best_k(result.results)
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz", "dunn"):
            metric_comparison_figure(result.results, metric, selected_k).write_html(
                FIGURES_DIR / f"{metric}_by_k_{experiment_signature[:12]}.html"
            )

    result = st.session_state.get("experiment_result")
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

        cards = st.columns(len(best))
        for column, (_, choice) in zip(cards, best.iterrows(), strict=True):
            with column:
                st.metric(
                    f"{choice['model']} — best K",
                    f"K = {int(choice['k'])}",
                    f"Silhouette {choice['silhouette']:.3f}",
                )
        st.caption(
            "K được chọn theo Silhouette cao nhất trong từng feature set; đây là kết quả "
            "đánh giá hình học của cụm, không phải kết luận về hiệu quả kinh doanh."
        )

        # ── 🏆 Best K summary comparison ──────────────────────────────
        st.divider()
        st.subheader("🏆 Kết quả tốt nhất – So sánh Best K giữa các model")
        st.caption(
            "Bảng tổng hợp tất cả internal metrics tại K tốt nhất của mỗi model. "
            "Ô màu xanh = metric tốt hơn, ô màu đỏ nhạt = metric kém hơn. "
            "Silhouette / Calinski-Harabasz / Dunn cao hơn = tốt hơn; Davies-Bouldin thấp hơn = tốt hơn."
        )

        comparison_rows = []
        for _, brow in best.iterrows():
            match = result.results[
                (result.results["model"] == brow["model"])
                & (result.results["k"] == brow["k"])
            ]
            if not match.empty:
                r = match.iloc[0]
                comparison_rows.append({
                    "Model": r["model"],
                    "Best K": int(r["k"]),
                    "Silhouette ↑": r.get("silhouette"),
                    "Davies-Bouldin ↓": r.get("davies_bouldin"),
                    "Calinski-Harabasz ↑": r.get("calinski_harabasz"),
                    "Dunn ↑": r.get("dunn"),
                    "Stability ARI (mean)": r.get("stability_ari_mean"),
                    "Stability ARI (std)": r.get("stability_ari_std"),
                })
        comparison = pd.DataFrame(comparison_rows)

        if len(comparison) >= 2:
            def highlight_winner(df: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                higher_better = ["Silhouette ↑", "Calinski-Harabasz ↑", "Dunn ↑", "Stability ARI (mean)"]
                lower_better = ["Davies-Bouldin ↓", "Stability ARI (std)"]
                for col in higher_better:
                    if col in df.columns:
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if vals.notna().sum() >= 2:
                            best_idx = vals.idxmax()
                            worst_idx = vals.idxmin()
                            if best_idx != worst_idx:
                                styles.loc[best_idx, col] = "background-color: #D1FAE5; font-weight: 700; color: #065F46"
                                styles.loc[worst_idx, col] = "background-color: #FEE2E2; color: #991B1B"
                for col in lower_better:
                    if col in df.columns:
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if vals.notna().sum() >= 2:
                            best_idx = vals.idxmin()
                            worst_idx = vals.idxmax()
                            if best_idx != worst_idx:
                                styles.loc[best_idx, col] = "background-color: #D1FAE5; font-weight: 700; color: #065F46"
                                styles.loc[worst_idx, col] = "background-color: #FEE2E2; color: #991B1B"
                for idx in df.index:
                    styles.loc[idx, "Best K"] = "font-weight: 700; background-color: #FDE68A; color: #92400E"
                return styles

            st.dataframe(
                comparison.style.apply(highlight_winner, axis=None),
                hide_index=True,
                use_container_width=True,
            )

            conclusions = []
            sil = pd.to_numeric(comparison["Silhouette ↑"], errors="coerce")
            if sil.notna().sum() >= 2:
                winner_idx = sil.idxmax()
                winner_model = comparison.loc[winner_idx, "Model"]
                winner_k = int(comparison.loc[winner_idx, "Best K"])
                winner_val = sil.loc[winner_idx]
                conclusions.append(
                    f"**{winner_model}** đạt Silhouette cao nhất ({winner_val:.4f}) tại **K = {winner_k}**"
                )
            db = pd.to_numeric(comparison["Davies-Bouldin ↓"], errors="coerce")
            if db.notna().sum() >= 2:
                winner_idx = db.idxmin()
                winner_model = comparison.loc[winner_idx, "Model"]
                winner_val = db.loc[winner_idx]
                conclusions.append(
                    f"**{winner_model}** đạt Davies-Bouldin thấp nhất ({winner_val:.4f})"
                )
            if conclusions:
                st.info("📊 " + " · ".join(conclusions))
        else:
            st.dataframe(comparison, hide_index=True, use_container_width=True)

        st.write("Chi tiết metrics tại Best K")
        detail_cols = st.columns(len(comparison))
        for col, (_, crow) in zip(detail_cols, comparison.iterrows()):
            with col:
                st.markdown(f"**{crow['Model']} · K = {int(crow['Best K'])}**")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    sil_val = crow.get("Silhouette ↑")
                    st.metric("Silhouette ↑", f"{sil_val:.4f}" if pd.notna(sil_val) else "N/A")
                    ch_val = crow.get("Calinski-Harabasz ↑")
                    st.metric("Calinski-Harabasz ↑", f"{ch_val:.1f}" if pd.notna(ch_val) else "N/A")
                with m_col2:
                    db_val = crow.get("Davies-Bouldin ↓")
                    st.metric("Davies-Bouldin ↓", f"{db_val:.4f}" if pd.notna(db_val) else "N/A")
                    dunn_val = crow.get("Dunn ↑")
                    st.metric("Dunn ↑", f"{dunn_val:.4f}" if pd.notna(dunn_val) else "N/A")
                ari_mean = crow.get("Stability ARI (mean)")
                ari_std = crow.get("Stability ARI (std)")
                if pd.notna(ari_mean):
                    st.metric("Stability ARI", f"{ari_mean:.4f} ± {ari_std:.4f}" if pd.notna(ari_std) else f"{ari_mean:.4f}")
        st.divider()

    st.write("Internal metric comparison")
    st.caption(
        "Internal metrics describe geometry, not business value. Metric scores depend on "
        "feature space; do not use Calinski–Harabasz alone to rank different feature sets."
    )
    metric_names = ("silhouette", "davies_bouldin", "calinski_harabasz", "dunn")
    best_keys = set(zip(best["model"], best["k"]))

    def highlight_selected_k(row: pd.Series) -> list[str]:
        selected = (row["model"], row["k"]) in best_keys
        if not selected:
            return [""] * len(row)
        styles = []
        for col in row.index:
            if col == "silhouette":
                styles.append(
                    "font-weight: 700; background-color: #BBF7D0; color: #065F46; "
                    "border-left: 4px solid #059669"
                )
            elif col == "k":
                styles.append(
                    "font-weight: 700; background-color: #FDE68A; color: #92400E; "
                    "border-left: 4px solid #F59E0B; font-size: 1.05em"
                )
            else:
                styles.append(
                    "font-weight: 700; background-color: #FEF3C7; color: #111827"
                )
        return styles

    st.write("Detailed results")
    styled = result.results.style.apply(highlight_selected_k, axis=1)
    styled = styled.apply(lambda df: _color_metrics_row(df), axis=None)
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption("Hàng nền vàng/xanh và chữ đậm là K được chọn theo Silhouette cao nhất. "
               "Gradient màu giúp so sánh trực quan giá trị metric giữa các K.")

    for metric in metric_names:
        st.plotly_chart(metric_comparison_figure(result.results, metric, best), use_container_width=True)

    selected_metric = st.selectbox("Metric chart to export as PNG", metric_names)
    metric_figure = metric_comparison_figure(result.results, selected_metric, best)
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
    choices = [(name, k) for name, k in result.labels]
    if choices:
        labels_display = [f"{name} · K={k}" for name, k in choices]
        preferred_choice = (best.iloc[0]["model"], int(best.iloc[0]["k"])) if not best.empty else choices[0]
        default_index = choices.index(preferred_choice) if preferred_choice in choices else 0
        selected_display = st.selectbox(
            "Profile and visualize a fitted model", labels_display, index=default_index
        )
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

        # ── 💾 Model Persistence: Save Pipeline Button ────────────────
        st.subheader("💾 Lưu Pipeline mô hình (Model Persistence)")
        st.caption(
            "Đóng gói quy trình tiền xử lý (Log1p -> StandardScaler) cùng mô hình KMeans đã huấn luyện "
            "vào file `.joblib` để sử dụng dự đoán tệp khách hàng mới tại Tab 'Dự đoán Khách hàng Mới'."
        )
        col_save_btn, col_save_status = st.columns([1, 2])
        with col_save_btn:
            if st.button(f"Lưu Pipeline [{choice[0]} · K={choice[1]}]", type="secondary"):
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                run_id = experiment_signature[:12]
                clean_name = choice[0].lower().replace(" ", "_")
                filename = f"{clean_name}_k{choice[1]}_{run_id}.joblib"
                target_path = MODELS_DIR / filename
                pipeline_obj = create_pipeline_from_experiment(
                    eligible_customer_features=result.eligible_customer_features,
                    features=st.session_state.experiment_models[choice[0]],
                    model_name=choice[0],
                    k=choice[1],
                    random_state=st.session_state.get("random_state", DEFAULT_RANDOM_STATE),
                    n_init=st.session_state.get("n_init", DEFAULT_N_INIT),
                    cluster_profiles=profile,
                    run_id=run_id,
                )
                save_pipeline(pipeline_obj, target_path)
                st.session_state[f"saved_pipe_{filename}"] = target_path
                with col_save_status:
                    st.success(f"✅ Đã lưu thành công pipeline vào: `results/models/{filename}`")

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
    ) if result.labels else pd.DataFrame(columns=["customer_id", "model", "k", "cluster"])
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


# ==============================================================================
# TAB 2: BUSINESS VALIDATION
# ==============================================================================
def render_business_validation_tab() -> None:
    st.header("💼 Xác thực Giá trị Kinh doanh (Business Validation)")
    st.caption(
        "Đánh giá kinh tế thực tế của các cụm khách hàng: Dự phóng CLV, Tỷ lệ giữ chân/tái mua, "
        "Kiểm định phân tách hành vi kinh tế giữa các mô hình, và Mô phỏng chiến dịch tiếp thị."
    )

    result = st.session_state.get("experiment_result")
    if result is None:
        st.info("💡 Chưa có kết quả thực nghiệm. Vui lòng hoàn thành bước phân cụm tại Tab 'Huấn luyện & Đánh giá' trước.")
        return

    best = select_best_k(result.results)
    if best.empty:
        st.warning("Chưa xác định được mô hình tốt nhất từ kết quả.")
        return

    # Choose model to analyze
    model_choices = list(result.scaled_features.keys())
    selected_model = st.selectbox("Chọn mô hình để xác thực kinh doanh", model_choices, index=len(model_choices) - 1)
    best_row = best.loc[best["model"] == selected_model]
    best_k = int(best_row.iloc[0]["k"]) if not best_row.empty else 2
    k_choice = st.number_input(f"Số cụm K ({selected_model})", min_value=2, max_value=10, value=best_k)

    key = (selected_model, int(k_choice))
    if key not in result.labels:
        st.error(f"Không tìm thấy nhãn phân cụm cho {selected_model} tại K={k_choice}.")
        return

    labels = result.labels[key]
    customer_features = result.eligible_customer_features

    # ── 1. Dự phóng CLV (Customer Lifetime Value) ────────────────────
    st.subheader("1. Dự phóng CLV (Customer Lifetime Value)")
    st.caption(
        r"CLV dự phóng cá nhân hóa theo công thức: "
        r"$\text{CLV} = M \times P(\text{Active}) \times \text{Profit Margin} \times \text{Time Horizon}$"
        r", trong đó $P(\text{Active}) = \exp(-\lambda \times R)$ suy giảm theo số ngày chưa mua hàng."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        margin = st.slider("Biên lợi nhuận gộp (Profit Margin %)", min_value=5, max_value=60, value=int(DEFAULT_PROFIT_MARGIN * 100)) / 100.0
    with c2:
        horizon = st.slider("Khung thời gian dự phóng (Năm)", min_value=0.5, max_value=3.0, value=DEFAULT_TIME_HORIZON_YEARS, step=0.5)
    with c3:
        half_life = st.number_input("Chu kỳ bán rã Recency (Ngày)", min_value=30, max_value=365, value=180)

    clv_summary = summarize_cluster_clv(
        customer_features,
        labels,
        selected_model,
        int(k_choice),
        profit_margin=margin,
        time_horizon_years=horizon,
    )
    st.dataframe(
        clv_summary.style.format({
            "mean_clv": "{:,.2f}",
            "median_clv": "{:,.2f}",
            "std_clv": "{:,.2f}",
            "total_projected_clv": "{:,.2f}",
            "p25_clv": "{:,.2f}",
            "p75_clv": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    fig_clv = px.bar(
        clv_summary,
        x="cluster",
        y="mean_clv",
        color="cluster",
        title=f"CLV Trung bình theo từng Cụm ({selected_model} · K={k_choice})",
        labels={"cluster": "Cụm khách hàng", "mean_clv": "CLV Dự phóng Trung bình"},
        text_auto=".2f",
    )
    st.plotly_chart(fig_clv, use_container_width=True)

    # ── 2. Tỷ lệ Giữ chân / Tái mua ──────────────────────────────────
    st.divider()
    st.subheader("2. Tỷ lệ Giữ chân & Tái mua (Retention & Repurchase Rate)")
    st.caption("Theo dõi khả năng duy trì quan hệ khách hàng và tần suất mua lặp lại của từng cụm.")
    recency_thresh = st.slider("Ngưỡng Recency để tính khách hàng đang hoạt động (Active Days)", 30, 180, DEFAULT_RETENTION_RECENCY_DAYS, 15)
    retention_df = calculate_retention_metrics(
        customer_features,
        labels,
        selected_model,
        int(k_choice),
        recency_threshold_days=recency_thresh,
    )
    st.dataframe(
        retention_df.style.format({
            "repurchase_rate": "{:.2%}",
            "high_frequency_rate": "{:.2%}",
            "active_rate": "{:.2%}",
            "median_recency_days": "{:.0f}",
            "median_frequency": "{:.1f}",
            "median_monetary": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    fig_ret = px.bar(
        retention_df,
        x="cluster",
        y=["repurchase_rate", "active_rate"],
        barmode="group",
        title=f"Tỷ lệ Tái mua và Khách hàng Đang hoạt động theo Cụm ({selected_model})",
        labels={"cluster": "Cụm", "value": "Tỷ lệ (%)", "variable": "Chỉ số"},
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── 3. So sánh Phân tách Kinh tế giữa các Mô hình ────────────────
    st.divider()
    st.subheader("3. Kiểm định Phân tách Kinh tế: Traditional RFM vs Extended RFM")
    st.caption(
        "Sử dụng phân tích phương sai One-Way ANOVA F-test trên các biến kinh tế thực tế. "
        "F-statistic càng cao chứng minh mô hình phân tách khách hàng theo giá trị kinh tế càng rõ nét."
    )
    if len(model_choices) >= 2 and len(best) >= 2:
        m1, m2 = model_choices[0], model_choices[1]
        k1 = int(best.loc[best["model"] == m1, "k"].iloc[0])
        k2 = int(best.loc[best["model"] == m2, "k"].iloc[0])
        sep_df = compare_business_separation(
            customer_features,
            result.labels[(m1, k1)],
            result.labels[(m2, k2)],
            profit_margin=margin,
        )
        st.dataframe(
            sep_df.style.format({
                "f_stat_traditional": "{:.2f}",
                "f_stat_extended": "{:.2f}",
                "p_val_traditional": "{:.4e}",
                "p_val_extended": "{:.4e}",
                "variance_ratio_diff_pct": "{:+.1f}%",
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Cần ít nhất 2 mô hình đã chạy thử nghiệm để so sánh phân tách kinh tế.")

    # ── 4. Mô phỏng Chiến dịch Marketing (Campaign Simulation) ──────
    st.divider()
    st.subheader("4. Mô phỏng Chiến dịch Marketing (Campaign Simulation)")
    st.caption(
        "Mô phỏng hiệu quả tài chính khi nhắm mục tiêu vào một cụm khách hàng cụ thể. "
        "Đặc biệt hữu ích để kiểm soát rủi ro biên lợi nhuận đối với nhóm Promotional Spend (PS) cao."
    )
    unique_clusters = sorted(list(np.unique(labels)))
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        target_cluster = st.selectbox("Chọn cụm khách hàng mục tiêu", unique_clusters, key="sim_target_cluster")
        discount_rate = st.slider("Mức chiết khấu / Khuyến mại (%)", min_value=5, max_value=35, value=10, step=1) / 100.0
        expected_response = st.slider("Tỷ lệ phản hồi dự kiến (Response Rate %)", min_value=5, max_value=50, value=15, step=5) / 100.0
    with sim_col2:
        mktg_cost = st.number_input("Chi phí marketing/liên hệ mỗi khách ($ / VNĐ)", min_value=0.1, value=1.0, step=0.5)
        lift_spend = st.slider("Tỷ lệ tăng trưởng doanh thu đơn hàng (Lift %)", min_value=0, max_value=50, value=10, step=5) / 100.0

    sim_res = simulate_promotional_campaign(
        customer_features,
        labels,
        target_cluster=int(target_cluster),
        discount_rate=discount_rate,
        expected_response_rate=expected_response,
        marketing_cost_per_customer=mktg_cost,
        baseline_margin=margin,
        lift_in_spend=lift_spend,
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Quy mô tiếp cận", f"{sim_res['target_audience_size']:,} khách", f"{sim_res['expected_responders']:,} phản hồi")
    with sc2:
        st.metric("Tổng chi phí chiến dịch", f"{sim_res['total_campaign_cost']:,.2f}")
    with sc3:
        st.metric("Doanh thu tạo thêm", f"{sim_res['incremental_revenue']:,.2f}")
    with sc4:
        st.metric("Lợi nhuận ròng / ROI", f"{sim_res['net_profit']:,.2f}", f"{sim_res['roi_pct']:+.1f}%")

    if "Cảnh báo" in sim_res["risk_assessment"]:
        st.warning(f"⚠️ {sim_res['risk_assessment']}")
    else:
        st.success(f"✅ {sim_res['risk_assessment']}")


# ==============================================================================
# TAB 3: BATCH INFERENCE & SCORING
# ==============================================================================
def render_inference_tab() -> None:
    st.header("🚀 Dự đoán Khách hàng Mới (Batch Inference & Scoring)")
    st.caption(
        "Sử dụng pipeline đã lưu (chuẩn hóa + K-Means) để phân cụm và gán nhãn hồ sơ cho danh sách khách hàng mới."
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved_list = list_saved_pipelines(MODELS_DIR)

    pipe_choice_mode = st.radio(
        "Chọn nguồn mô hình:",
        ["Chọn từ thư mục kết quả (results/models/)", "Tải lên file pipeline (.joblib)"],
        horizontal=True,
    )

    pipeline: SegmentationPipeline | None = None
    if pipe_choice_mode == "Chọn từ thư mục kết quả (results/models/)":
        if not saved_list:
            st.warning("Chưa có pipeline nào được lưu trong `results/models/`. Hãy huấn luyện mô hình ở Tab 1 và nhấn 'Lưu Pipeline'.")
        else:
            opts = [f"{p['filename']} ({p['model_name']} · K={p['k']})" for p in saved_list]
            selected_idx = st.selectbox("Chọn pipeline:", range(len(opts)), format_func=lambda i: opts[i])
            chosen_path = saved_list[selected_idx]["path"]
            try:
                pipeline = load_pipeline(chosen_path)
            except Exception as e:
                st.error(f"Lỗi nạp pipeline: {e}")
    else:
        uploaded_pipe = st.file_uploader("Tải lên file .joblib", type=["joblib"])
        if uploaded_pipe is not None:
            tmp_pipe_path = MODELS_DIR / f"_temp_{uploaded_pipe.name}"
            tmp_pipe_path.write_bytes(uploaded_pipe.getvalue())
            try:
                pipeline = load_pipeline(tmp_pipe_path)
            except Exception as e:
                st.error(f"Không thể đọc file pipeline: {e}")

    if pipeline is not None:
        st.info(
            f"📦 **Pipeline đang chọn:** `{pipeline.model_name}` | "
            f"**Số cụm K:** `{pipeline.k}` | "
            f"**Đặc trưng yêu cầu:** `{', '.join(pipeline.features)}` | "
            f"**Thời gian lưu:** `{pipeline.created_at[:19]}`"
        )

        st.subheader("Tải lên dữ liệu khách hàng mới")
        new_data_file = st.file_uploader("Tải lên danh sách khách hàng mới (CSV / XLSX)", type=["csv", "xlsx"], key="inference_data_uploader")
        if new_data_file is None:
            st.info("Vui lòng tải lên file dữ liệu khách hàng mới để tiến hành phân cụm.")
            return

        try:
            new_df = read_uploaded_file(new_data_file.getvalue(), new_data_file.name)
        except Exception as e:
            st.error(f"Không thể đọc file: {e}")
            return

        new_df.columns = new_df.columns.map(str)
        st.write(f"Tập dữ liệu tải lên: {len(new_df):,} dòng × {len(new_df.columns)} cột")
        st.dataframe(new_df.head(10), use_container_width=True)

        # Check if table already has features or is raw transactions
        has_all_features = all(f in new_df.columns for f in pipeline.features)
        if has_all_features:
            customer_table = new_df.copy()
            st.success("✅ Dữ liệu đã có sẵn đầy đủ các cột đặc trưng mô hình yêu cầu!")
        else:
            st.warning("Dữ liệu chưa có sẵn đủ các cột đặc trưng. Nếu đây là dữ liệu giao dịch thô, vui lòng ánh xạ cột để tính toán:")
            cols = list(new_df.columns)
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                cid = _column_selector("Customer ID (required)", cols, True, key_prefix="inf")
                oid = _column_selector("Order ID (required)", cols, True, key_prefix="inf")
                pdate = _column_selector("Purchase Date (required)", cols, True, key_prefix="inf")
            with m_c2:
                sp = _column_selector("Spend Amount (required)", cols, True, key_prefix="inf")
                cat = _column_selector("Category (for D)", cols, False, key_prefix="inf")
                promo = _column_selector("Promo Spend (for PS)", cols, False, key_prefix="inf")

            mapping_inf = ColumnMapping(cid, oid, pdate, sp, cat, promo)
            try:
                cleaned_inf, _ = clean_transactions(new_df, mapping_inf)
                customer_table, _ = build_customer_features(cleaned_inf, mapping_inf)
                st.success(f"✅ Đã trích xuất đặc trưng cho {len(customer_table):,} khách hàng.")
            except Exception as e:
                st.error(f"Lỗi xử lý giao dịch: {e}")
                return

        if st.button("🚀 Phân cụm khách hàng mới (Batch Scoring)", type="primary"):
            try:
                scored = predict_customer_segments(pipeline, customer_table)
            except Exception as e:
                st.error(f"Lỗi dự đoán phân cụm: {e}")
                return

            st.subheader("Kết quả Phân cụm Khách hàng Mới")
            assigned_only = scored[scored["status"] == "assigned"]
            st.caption(f"Đã gán nhãn thành công cho {len(assigned_only):,} / {len(scored):,} khách hàng.")

            if not assigned_only.empty:
                # Distribution chart
                cluster_counts = assigned_only["predicted_cluster"].value_counts().reset_index()
                cluster_counts.columns = ["cluster", "count"]
                cluster_counts["cluster"] = cluster_counts["cluster"].astype(int).astype(str)

                fig_dist = px.bar(
                    cluster_counts,
                    x="cluster",
                    y="count",
                    title="Phân bổ Khách hàng Mới theo Cụm",
                    labels={"cluster": "Cụm khách hàng", "count": "Số lượng khách"},
                    color="cluster",
                    text_auto=True,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            st.dataframe(scored, use_container_width=True)

            # Download CSV
            scored_csv = scored.to_csv(index=True).encode("utf-8")
            st.download_button(
                "Tải xuống kết quả phân cụm (CSV)",
                data=scored_csv,
                file_name=f"new_customer_segments_{pipeline.model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main() -> None:
    st.set_page_config(page_title="Extended RFM Research", layout="wide")
    st.title("Extended RFM Customer Segmentation")
    st.caption(
        "Nghiên cứu & Ứng dụng: Phân khúc khách hàng đa chiều bằng K-Means trên RFM Mở rộng (R, F, M, D, PS, PR), "
        "kết hợp kiểm chứng hình học, xác thực giá trị kinh doanh và chấm điểm khách hàng mới."
    )

    tab_train, tab_business, tab_inference = st.tabs([
        "🔬 Huấn luyện & Đánh giá (Train & Validate)",
        "💼 Xác thực Giá trị Kinh doanh (Business Validation)",
        "🚀 Dự đoán Khách hàng Mới (Batch Inference)",
    ])

    with tab_train:
        render_training_tab()

    with tab_business:
        render_business_validation_tab()

    with tab_inference:
        render_inference_tab()


if __name__ == "__main__":
    main()
