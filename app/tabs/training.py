"""Tab 1: Training, Model Comparison, and Geometric Validation."""

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui_common import (
    FIGURES_DIR,
    MODELS_DIR,
    TABLES_DIR,
    color_metrics_row,
    column_selector,
    highlight_selected_k,
    read_uploaded_file,
)
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
from src.pipeline_manager import create_pipeline_from_experiment, save_pipeline
from src.preprocessing import clean_transactions, explore_dataset
from src.visualization import (
    cluster_scatter_figure,
    metric_comparison_figure,
    profile_clusters,
)


def render_training_tab() -> None:
    """Render the full training, evaluation, and pipeline export tab."""
    with st.expander("Research definitions and assumptions", expanded=False):
        st.markdown(
            "- Traditional RFM = R, F, M. Configured Extended RFM = R, F, M, D, PS, PR.\n"
            "- F counts unique invoices, not transaction rows. M sums the mapped spend field.\n"
            "- PS requires a verified promotional-spending amount, not an assumed discount.\n"
            "- PR is missing for customers with fewer than three distinct purchase dates.\n"
            "- `log1p` and StandardScaler are applied consistently; selected-feature gaps "
            "exclude the same customers from every model."
        )

    uploaded = st.file_uploader(
        "Upload transaction data", type=["csv", "xlsx"], key="train_uploader"
    )
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
        customer_id = column_selector("Customer ID (required)", columns, True)
        order_id = column_selector("Order / invoice ID (required)", columns, True)
        purchase_date = column_selector("Purchase date (required)", columns, True)
    with right:
        spend = column_selector("Transaction spend (required)", columns, True)
        category = column_selector("Product category (for D)", columns, False)
        promotional_spend = column_selector(
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

        # ──  Best K summary comparison ──────────────────────────────
        st.divider()
        st.subheader("Kết quả tốt nhất – So sánh Best K giữa các model")
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
                comparison_rows.append(
                    {
                        "Model": r["model"],
                        "Best K": int(r["k"]),
                        "Silhouette ↑": r.get("silhouette"),
                        "Davies-Bouldin ↓": r.get("davies_bouldin"),
                        "Calinski-Harabasz ↑": r.get("calinski_harabasz"),
                        "Dunn ↑": r.get("dunn"),
                        "Stability ARI (mean)": r.get("stability_ari_mean"),
                        "Stability ARI (std)": r.get("stability_ari_std"),
                    }
                )
        comparison = pd.DataFrame(comparison_rows)

        if len(comparison) >= 2:

            def highlight_winner(df: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                higher_better = [
                    "Silhouette ↑",
                    "Calinski-Harabasz ↑",
                    "Dunn ↑",
                    "Stability ARI (mean)",
                ]
                lower_better = ["Davies-Bouldin ↓", "Stability ARI (std)"]
                for col in higher_better:
                    if col in df.columns:
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if vals.notna().sum() >= 2:
                            best_idx = vals.idxmax()
                            worst_idx = vals.idxmin()
                            if best_idx != worst_idx:
                                styles.loc[best_idx, col] = (
                                    "background-color: #D1FAE5; font-weight: 700; color: #065F46"
                                )
                                styles.loc[worst_idx, col] = (
                                    "background-color: #FEE2E2; color: #991B1B"
                                )
                for col in lower_better:
                    if col in df.columns:
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if vals.notna().sum() >= 2:
                            best_idx = vals.idxmin()
                            worst_idx = vals.idxmax()
                            if best_idx != worst_idx:
                                styles.loc[best_idx, col] = (
                                    "background-color: #D1FAE5; font-weight: 700; color: #065F46"
                                )
                                styles.loc[worst_idx, col] = (
                                    "background-color: #FEE2E2; color: #991B1B"
                                )
                for idx in df.index:
                    styles.loc[idx, "Best K"] = (
                        "font-weight: 700; background-color: #FDE68A; color: #92400E"
                    )
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
                st.info(" · ".join(conclusions))
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
                    st.metric(
                        "Calinski-Harabasz ↑", f"{ch_val:.1f}" if pd.notna(ch_val) else "N/A"
                    )
                with m_col2:
                    db_val = crow.get("Davies-Bouldin ↓")
                    st.metric("Davies-Bouldin ↓", f"{db_val:.4f}" if pd.notna(db_val) else "N/A")
                    dunn_val = crow.get("Dunn ↑")
                    st.metric("Dunn ↑", f"{dunn_val:.4f}" if pd.notna(dunn_val) else "N/A")
                ari_mean = crow.get("Stability ARI (mean)")
                ari_std = crow.get("Stability ARI (std)")
                if pd.notna(ari_mean):
                    st.metric(
                        "Stability ARI",
                        f"{ari_mean:.4f} ± {ari_std:.4f}" if pd.notna(ari_std) else f"{ari_mean:.4f}",
                    )
        st.divider()

    st.write("Internal metric comparison")
    st.caption(
        "Internal metrics describe geometry, not business value. Metric scores depend on "
        "feature space; do not use Calinski–Harabasz alone to rank different feature sets."
    )
    metric_names = ("silhouette", "davies_bouldin", "calinski_harabasz", "dunn")
    best_keys = set(zip(best["model"], best["k"]))

    st.write("Detailed results")
    styled = result.results.style.apply(lambda r: highlight_selected_k(r, best_keys), axis=1)
    styled = styled.apply(lambda df: color_metrics_row(df), axis=None)
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption(
        "Hàng nền vàng/xanh và chữ đậm là K được chọn theo Silhouette cao nhất. "
        "Gradient màu giúp so sánh trực quan giá trị metric giữa các K."
    )

    for metric in metric_names:
        st.plotly_chart(
            metric_comparison_figure(result.results, metric, best), use_container_width=True
        )

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
        preferred_choice = (
            (best.iloc[0]["model"], int(best.iloc[0]["k"])) if not best.empty else choices[0]
        )
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

        # ── Model Persistence: Save Pipeline Button ────────────────
        st.subheader("Lưu Pipeline mô hình (Model Persistence)")
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
                    st.success(f"Đã lưu thành công pipeline vào: `results/models/{filename}`")

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
        all_profiles.append(profile_clusters(result.eligible_customer_features, labels, model, k))
    profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    profiles_csv = profiles.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cluster profiles (CSV)",
        data=profiles_csv,
        file_name="cluster_profiles.csv",
        mime="text/csv",
    )
    assignments = (
        pd.concat(
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
        )
        if result.labels
        else pd.DataFrame(columns=["customer_id", "model", "k", "cluster"])
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
