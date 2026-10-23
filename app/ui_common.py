"""Shared UI constants, file readers, column pickers, and styling helpers."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
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


def column_selector(
    label: str,
    columns: list[str],
    required: bool,
    key_prefix: str = "mapping",
) -> str | None:
    """Render a selectbox for mapping dataset columns."""
    options = columns if required else ["(not mapped)"] + columns
    selected = st.selectbox(label, options=options, key=f"{key_prefix}_{label}")
    return selected if required or selected != "(not mapped)" else None


def css_gradient(
    series: pd.Series,
    low_rgb: tuple[int, int, int],
    high_rgb: tuple[int, int, int],
    reverse: bool = False,
) -> list[str]:
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


def color_metrics_row(df: pd.DataFrame) -> pd.DataFrame:
    """Apply gradient coloring to metric columns without matplotlib."""
    out = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in ("silhouette", "dunn"):
        if col in df.columns:
            out[col] = css_gradient(df[col], (255, 255, 255), (187, 247, 208))
    if "davies_bouldin" in df.columns:
        out["davies_bouldin"] = css_gradient(
            df["davies_bouldin"], (255, 255, 255), (254, 226, 226), reverse=True
        )
    if "calinski_harabasz" in df.columns:
        out["calinski_harabasz"] = css_gradient(
            df["calinski_harabasz"], (255, 255, 255), (191, 219, 254)
        )
    return out


def highlight_selected_k(row: pd.Series, best_keys: set[tuple[str, int]]) -> list[str]:
    """Highlight the best-K row with strong background and distinct borders."""
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
            styles.append("font-weight: 700; background-color: #FEF3C7; color: #111827")
    return styles
