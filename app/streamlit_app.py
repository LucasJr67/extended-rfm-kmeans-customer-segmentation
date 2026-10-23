"""Interactive research dashboard for Extended RFM Customer Segmentation.

Main application entrypoint delegating to modular tabs:
1. Tab 1: Training, Model Comparison, and Geometric Validation (app/tabs/training.py)
2. Tab 2: Business Validation & Economic Metrics (app/tabs/business.py)
3. Tab 3: Batch Customer Inference & Scoring (app/tabs/inference.py)
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.tabs.business import render_business_validation_tab
from app.tabs.inference import render_inference_tab
from app.tabs.training import render_training_tab


def main() -> None:
    """Configure dashboard and render navigation tabs."""
    st.set_page_config(page_title="Extended RFM Research", layout="wide")
    st.title("Extended RFM Customer Segmentation")
    st.caption(
        "Nghiên cứu & Ứng dụng: Phân khúc khách hàng đa chiều bằng K-Means trên RFM Mở rộng (R, F, M, D, PS, PR), "
        "kết hợp kiểm chứng hình học, xác thực giá trị kinh doanh và chấm điểm khách hàng mới."
    )

    tab_train, tab_business, tab_inference = st.tabs(
        [
            "Huấn luyện & Đánh giá (Train & Validate)",
            "Xác thực Giá trị Kinh doanh (Business Validation)",
            "Dự đoán Khách hàng Mới (Batch Inference)",
        ]
    )

    with tab_train:
        render_training_tab()

    with tab_business:
        render_business_validation_tab()

    with tab_inference:
        render_inference_tab()


if __name__ == "__main__":
    main()
