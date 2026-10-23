"""Business validation and economic impact analysis for customer segmentation."""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    DEFAULT_PROFIT_MARGIN,
    DEFAULT_RETENTION_RECENCY_DAYS,
    DEFAULT_TIME_HORIZON_YEARS,
)


def calculate_customer_clv(
    customer_features: pd.DataFrame,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    time_horizon_years: float = DEFAULT_TIME_HORIZON_YEARS,
    recency_half_life_days: float = 180.0,
) -> pd.Series:
    """Project heuristic Customer Lifetime Value (CLV) at the individual customer level.

    Formula:
        AOV = M / F (for F >= 1, M > 0)
        P_active = exp(-ln(2) * R / recency_half_life_days)
        Expected_Annual_Spend = M * P_active
        CLV = Expected_Annual_Spend * profit_margin * time_horizon_years

    Returns a Series of CLV values indexed by customer ID.
    """
    if "R" not in customer_features or "F" not in customer_features or "M" not in customer_features:
        raise ValueError("Customer features must include 'R', 'F', and 'M'.")
    if profit_margin < 0 or profit_margin > 1:
        raise ValueError("Profit margin must be between 0 and 1.")
    if time_horizon_years <= 0:
        raise ValueError("Time horizon must be positive.")

    r = customer_features["R"].astype(float).clip(lower=0)
    m = customer_features["M"].astype(float).clip(lower=0)

    # Activity probability decays with recency
    decay_rate = np.log(2) / max(recency_half_life_days, 1.0)
    p_active = np.exp(-decay_rate * r)

    annual_spend = m * p_active
    clv = annual_spend * profit_margin * time_horizon_years
    return pd.Series(clv, index=customer_features.index, name="projected_clv")


def summarize_cluster_clv(
    customer_features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    k: int,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    time_horizon_years: float = DEFAULT_TIME_HORIZON_YEARS,
) -> pd.DataFrame:
    """Aggregate CLV statistics by cluster for a given clustering model."""
    aligned_labels = labels.loc[labels.index.intersection(customer_features.index)]
    clv_series = calculate_customer_clv(
        customer_features.loc[aligned_labels.index],
        profit_margin=profit_margin,
        time_horizon_years=time_horizon_years,
    )
    df = pd.DataFrame({"cluster": aligned_labels, "clv": clv_series})

    rows = []
    for cluster_id, group in df.groupby("cluster"):
        vals = group["clv"]
        rows.append(
            {
                "model": model_name,
                "k": k,
                "cluster": int(cluster_id),
                "customer_count": int(len(vals)),
                "mean_clv": float(vals.mean()),
                "median_clv": float(vals.median()),
                "std_clv": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "total_projected_clv": float(vals.sum()),
                "p25_clv": float(vals.quantile(0.25)),
                "p75_clv": float(vals.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def calculate_retention_metrics(
    customer_features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    k: int,
    recency_threshold_days: int = DEFAULT_RETENTION_RECENCY_DAYS,
) -> pd.DataFrame:
    """Compute retention and repurchase indicators across clusters.

    Indicators:
        - Repurchase rate: proportion of customers with Frequency >= 2.
        - High frequency rate: proportion of customers with Frequency >= 3.
        - Active customer rate: proportion with Recency <= recency_threshold_days.
    """
    if "R" not in customer_features or "F" not in customer_features:
        raise ValueError("Customer features must include 'R' and 'F'.")
    common_idx = customer_features.index.intersection(labels.index)
    cf = customer_features.loc[common_idx]
    y = labels.loc[common_idx]

    rows = []
    for cluster_id, group_idx in y.groupby(y).groups.items():
        sub = cf.loc[group_idx]
        n = len(sub)
        repurchase_count = int((sub["F"] >= 2).sum())
        high_freq_count = int((sub["F"] >= 3).sum())
        active_count = int((sub["R"] <= recency_threshold_days).sum())

        rows.append(
            {
                "model": model_name,
                "k": k,
                "cluster": int(cluster_id),
                "customer_count": n,
                "repurchase_count": repurchase_count,
                "repurchase_rate": float(repurchase_count / n) if n else 0.0,
                "high_frequency_count": high_freq_count,
                "high_frequency_rate": float(high_freq_count / n) if n else 0.0,
                "active_customer_count": active_count,
                "active_rate": float(active_count / n) if n else 0.0,
                "median_recency_days": float(sub["R"].median()),
                "median_frequency": float(sub["F"].median()),
                "median_monetary": float(sub["M"].median()) if "M" in sub else 0.0,
            }
        )
    return pd.DataFrame(rows)


def simulate_promotional_campaign(
    customer_features: pd.DataFrame,
    labels: pd.Series,
    target_cluster: int,
    discount_rate: float = 0.10,
    expected_response_rate: float = 0.15,
    marketing_cost_per_customer: float = 1.0,
    baseline_margin: float = 0.20,
    lift_in_spend: float = 0.10,
) -> dict[str, Any]:
    """Simulate marketing campaign economics targeting a specific customer cluster.

    Estimates campaign expenses, incremental sales, net margin, and ROI.
    Flags promotional sensitivity risk if PS is high or ROI is negative.
    """
    common_idx = customer_features.index.intersection(labels.index)
    cf = customer_features.loc[common_idx]
    y = labels.loc[common_idx]

    cluster_members = cf.loc[y == target_cluster]
    n_target = len(cluster_members)
    if n_target == 0:
        raise ValueError(f"Cluster {target_cluster} has no customers.")

    mean_spend = float(cluster_members["M"].mean())
    mean_f = float(cluster_members["F"].mean())
    aov = mean_spend / max(mean_f, 1.0)
    median_ps = float(cluster_members["PS"].median()) if "PS" in cluster_members and cluster_members["PS"].notna().any() else None

    # Responders
    n_responders = int(round(n_target * expected_response_rate))
    n_responders = max(1, min(n_responders, n_target))

    # Costs
    contact_cost = float(n_target * marketing_cost_per_customer)
    post_lift_order_spend = aov * (1.0 + lift_in_spend)
    discount_cost_per_responder = post_lift_order_spend * discount_rate
    total_discount_cost = float(n_responders * discount_cost_per_responder)
    total_cost = contact_cost + total_discount_cost

    # Revenues & Margin
    incremental_gross_revenue = float(n_responders * post_lift_order_spend)
    gross_margin_earned = incremental_gross_revenue * baseline_margin
    net_profit = float(gross_margin_earned - total_cost)
    roi_pct = float((net_profit / total_cost) * 100.0) if total_cost > 0 else 0.0

    # Risk Assessment
    risk_notes = []
    if median_ps is not None and median_ps > 0.35:
        risk_notes.append(
            "Cảnh báo: Cụm này có tỷ lệ PS (chi tiêu khuyến mãi) trung vị cao (>35%). "
            "Khách hàng có xu hướng săn deal, chiết khấu sâu có nguy cơ làm xói mòn biên lợi nhuận."
        )
    if net_profit < 0:
        risk_notes.append(
            "Cảnh báo ROI âm: Chi phí marketing và chiết khấu vượt quá lợi nhuận gộp tạo thêm. "
            "Cần tăng điều kiện giá trị đơn hàng tối thiểu (Min Order Value) hoặc giảm chiết khấu."
        )
    if not risk_notes:
        risk_notes.append("Chiến dịch khả thi: ROI dương và biên lợi nhuận được đảm bảo.")

    return {
        "target_cluster": int(target_cluster),
        "target_audience_size": n_target,
        "expected_responders": n_responders,
        "aov_baseline": aov,
        "post_lift_aov": post_lift_order_spend,
        "marketing_contact_cost": contact_cost,
        "total_discount_cost": total_discount_cost,
        "total_campaign_cost": total_cost,
        "incremental_revenue": incremental_gross_revenue,
        "gross_margin_earned": gross_margin_earned,
        "net_profit": net_profit,
        "roi_pct": roi_pct,
        "median_ps": median_ps,
        "risk_assessment": " ".join(risk_notes),
    }


def compare_business_separation(
    customer_features: pd.DataFrame,
    labels_traditional: pd.Series,
    labels_extended: pd.Series,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
) -> pd.DataFrame:
    """Compare how cleanly Traditional RFM vs Extended RFM separate key economic metrics.

    Uses One-Way ANOVA F-statistic: higher F indicates stronger between-cluster separation.
    """
    common_idx = customer_features.index.intersection(labels_traditional.index).intersection(labels_extended.index)
    cf = customer_features.loc[common_idx]
    y_trad = labels_traditional.loc[common_idx]
    y_ext = labels_extended.loc[common_idx]

    clv = calculate_customer_clv(cf, profit_margin=profit_margin)
    cf_with_clv = cf.copy()
    cf_with_clv["projected_clv"] = clv
    cf_with_clv["aov"] = cf_with_clv["M"] / cf_with_clv["F"].clip(lower=1)

    metrics_to_test = ["projected_clv", "M", "F", "aov"]
    rows = []
    for col in metrics_to_test:
        vals = cf_with_clv[col].to_numpy()

        # Traditional F-test
        trad_groups = [vals[y_trad == c] for c in np.unique(y_trad) if len(vals[y_trad == c]) > 1]
        if len(trad_groups) >= 2:
            f_trad, p_trad = stats.f_oneway(*trad_groups)
        else:
            f_trad, p_trad = 0.0, 1.0

        # Extended F-test
        ext_groups = [vals[y_ext == c] for c in np.unique(y_ext) if len(vals[y_ext == c]) > 1]
        if len(ext_groups) >= 2:
            f_ext, p_ext = stats.f_oneway(*ext_groups)
        else:
            f_ext, p_ext = 0.0, 1.0

        rows.append(
            {
                "economic_metric": col,
                "f_stat_traditional": float(f_trad),
                "p_val_traditional": float(p_trad),
                "f_stat_extended": float(f_ext),
                "p_val_extended": float(p_ext),
                "better_separation": "Extended RFM" if f_ext > f_trad else "Traditional RFM",
                "variance_ratio_diff_pct": float(((f_ext - f_trad) / max(f_trad, 1e-6)) * 100.0),
            }
        )
    return pd.DataFrame(rows)
