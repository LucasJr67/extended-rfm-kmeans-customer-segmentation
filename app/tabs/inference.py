"""Tab 3: Batch Customer Inference & Scoring."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_common import MODELS_DIR, column_selector, read_uploaded_file
from src.config import ColumnMapping
from src.features import build_customer_features
from src.pipeline_manager import (
    SegmentationPipeline,
    list_saved_pipelines,
    load_pipeline,
    predict_customer_segments,
)
from src.preprocessing import clean_transactions


def render_inference_tab() -> None:
    """Render pipeline selection, new customer upload, and batch segment prediction."""
    st.header("Dự đoán Khách hàng Mới (Batch Inference & Scoring)")
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
            st.warning(
                "Chưa có pipeline nào được lưu trong `results/models/`. "
                "Hãy huấn luyện mô hình ở Tab 1 và nhấn 'Lưu Pipeline'."
            )
        else:
            opts = [f"{p['filename']} ({p['model_name']} · K={p['k']})" for p in saved_list]
            selected_idx = st.selectbox(
                "Chọn pipeline:", range(len(opts)), format_func=lambda i: opts[i]
            )
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
            f"**Pipeline đang chọn:** `{pipeline.model_name}` | "
            f"**Số cụm K:** `{pipeline.k}` | "
            f"**Đặc trưng yêu cầu:** `{', '.join(pipeline.features)}` | "
            f"**Thời gian lưu:** `{pipeline.created_at[:19]}`"
        )

        st.subheader("Tải lên dữ liệu khách hàng mới")
        new_data_file = st.file_uploader(
            "Tải lên danh sách khách hàng mới (CSV / XLSX)",
            type=["csv", "xlsx"],
            key="inference_data_uploader",
        )
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
            st.success("Dữ liệu đã có sẵn đầy đủ các cột đặc trưng mô hình yêu cầu!")
        else:
            st.warning(
                "Dữ liệu chưa có sẵn đủ các cột đặc trưng. Nếu đây là dữ liệu giao dịch thô, "
                "vui lòng ánh xạ cột để tính toán:"
            )
            cols = list(new_df.columns)
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                cid = column_selector("Customer ID (required)", cols, True, key_prefix="inf")
                oid = column_selector("Order ID (required)", cols, True, key_prefix="inf")
                pdate = column_selector("Purchase Date (required)", cols, True, key_prefix="inf")
            with m_c2:
                sp = column_selector("Spend Amount (required)", cols, True, key_prefix="inf")
                cat = column_selector("Category (for D)", cols, False, key_prefix="inf")
                promo = column_selector("Promo Spend (for PS)", cols, False, key_prefix="inf")

            mapping_inf = ColumnMapping(cid, oid, pdate, sp, cat, promo)
            try:
                cleaned_inf, _ = clean_transactions(new_df, mapping_inf)
                customer_table, _ = build_customer_features(cleaned_inf, mapping_inf)
                st.success(f"Đã trích xuất đặc trưng cho {len(customer_table):,} khách hàng.")
            except Exception as e:
                st.error(f"Lỗi xử lý giao dịch: {e}")
                return

        if st.button("Phân cụm khách hàng mới (Batch Scoring)", type="primary"):
            try:
                scored = predict_customer_segments(pipeline, customer_table)
            except Exception as e:
                st.error(f"Lỗi dự đoán phân cụm: {e}")
                return

            st.subheader("Kết quả Phân cụm Khách hàng Mới")
            assigned_only = scored[scored["status"] == "assigned"]
            st.caption(
                f"Đã gán nhãn thành công cho {len(assigned_only):,} / {len(scored):,} khách hàng."
            )

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
