"""Tab 2: Business Validation & Economic Metrics."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.business_validation import (
    calculate_customer_clv,
    calculate_retention_metrics,
    compare_business_separation,
    simulate_promotional_campaign,
    summarize_cluster_clv,
)
from src.clustering import select_best_k
from src.config import (
    DEFAULT_PROFIT_MARGIN,
    DEFAULT_RETENTION_RECENCY_DAYS,
    DEFAULT_TIME_HORIZON_YEARS,
)


def render_business_validation_tab() -> None:
    """Render CLV projection, retention rates, ANOVA separation, and campaign simulation."""
    st.header("Xác thực Giá trị Kinh doanh (Business Validation)")
    st.caption(
        "Đánh giá kinh tế thực tế của các cụm khách hàng: Dự phóng CLV, Tỷ lệ giữ chân/tái mua, "
        "Kiểm định phân tách hành vi kinh tế giữa các mô hình, và Mô phỏng chiến dịch tiếp thị."
    )

    result = st.session_state.get("experiment_result")
    if result is None:
        st.info(
            "💡 Chưa có kết quả thực nghiệm. Vui lòng hoàn thành bước phân cụm tại Tab 'Huấn luyện & Đánh giá' trước."
        )
        return

    best = select_best_k(result.results)
    if best.empty:
        st.warning("Chưa xác định được mô hình tốt nhất từ kết quả.")
        return

    # Choose model to analyze
    model_choices = list(result.scaled_features.keys())
    selected_model = st.selectbox(
        "Chọn mô hình để xác thực kinh doanh", model_choices, index=len(model_choices) - 1
    )
    best_row = best.loc[best["model"] == selected_model]
    best_k = int(best_row.iloc[0]["k"]) if not best_row.empty else 2
    k_choice = st.number_input(
        f"Số cụm K ({selected_model})", min_value=2, max_value=10, value=best_k
    )

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
        margin = (
            st.slider(
                "Biên lợi nhuận gộp (Profit Margin %)",
                min_value=5,
                max_value=60,
                value=int(DEFAULT_PROFIT_MARGIN * 100),
            )
            / 100.0
        )
    with c2:
        horizon = st.slider(
            "Khung thời gian dự phóng (Năm)",
            min_value=0.5,
            max_value=3.0,
            value=DEFAULT_TIME_HORIZON_YEARS,
            step=0.5,
        )
    with c3:
        half_life = st.number_input(
            "Chu kỳ bán rã Recency (Ngày)", min_value=30, max_value=365, value=180
        )

    clv_summary = summarize_cluster_clv(
        customer_features,
        labels,
        selected_model,
        int(k_choice),
        profit_margin=margin,
        time_horizon_years=horizon,
    )
    st.dataframe(
        clv_summary.style.format(
            {
                "mean_clv": "{:,.2f}",
                "median_clv": "{:,.2f}",
                "std_clv": "{:,.2f}",
                "total_projected_clv": "{:,.2f}",
                "p25_clv": "{:,.2f}",
                "p75_clv": "{:,.2f}",
            }
        ),
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
    recency_thresh = st.slider(
        "Ngưỡng Recency để tính khách hàng đang hoạt động (Active Days)",
        30,
        180,
        DEFAULT_RETENTION_RECENCY_DAYS,
        15,
    )
    retention_df = calculate_retention_metrics(
        customer_features,
        labels,
        selected_model,
        int(k_choice),
        recency_threshold_days=recency_thresh,
    )
    st.dataframe(
        retention_df.style.format(
            {
                "repurchase_rate": "{:.2%}",
                "high_frequency_rate": "{:.2%}",
                "active_rate": "{:.2%}",
                "median_recency_days": "{:.0f}",
                "median_frequency": "{:.1f}",
                "median_monetary": "{:,.2f}",
            }
        ),
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
            sep_df.style.format(
                {
                    "f_stat_traditional": "{:.2f}",
                    "f_stat_extended": "{:.2f}",
                    "p_val_traditional": "{:.4e}",
                    "p_val_extended": "{:.4e}",
                    "variance_ratio_diff_pct": "{:+.1f}%",
                }
            ),
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
        target_cluster = st.selectbox(
            "Chọn cụm khách hàng mục tiêu", unique_clusters, key="sim_target_cluster"
        )
        discount_rate = (
            st.slider(
                "Mức chiết khấu / Khuyến mại (%)", min_value=5, max_value=35, value=10, step=1
            )
            / 100.0
        )
        expected_response = (
            st.slider(
                "Tỷ lệ phản hồi dự kiến (Response Rate %)",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
            )
            / 100.0
        )
    with sim_col2:
        mktg_cost = st.number_input(
            "Chi phí marketing/liên hệ mỗi khách ($ / VNĐ)", min_value=0.1, value=1.0, step=0.5
        )
        lift_spend = (
            st.slider(
                "Tỷ lệ tăng trưởng doanh thu đơn hàng (Lift %)",
                min_value=0,
                max_value=50,
                value=10,
                step=5,
            )
            / 100.0
        )

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
        st.metric(
            "Quy mô tiếp cận",
            f"{sim_res['target_audience_size']:,} khách",
            f"{sim_res['expected_responders']:,} phản hồi",
        )
    with sc2:
        st.metric("Tổng chi phí chiến dịch", f"{sim_res['total_campaign_cost']:,.2f}")
    with sc3:
        st.metric("Doanh thu tạo thêm", f"{sim_res['incremental_revenue']:,.2f}")
    with sc4:
        st.metric("Lợi nhuận ròng / ROI", f"{sim_res['net_profit']:,.2f}", f"{sim_res['roi_pct']:+.1f}%")

    if "Cảnh báo" in sim_res["risk_assessment"]:
        st.warning(f"{sim_res['risk_assessment']}")
    else:
        st.success(f"{sim_res['risk_assessment']}")
