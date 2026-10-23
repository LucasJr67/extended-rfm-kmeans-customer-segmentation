"""Tabs package for the Streamlit research dashboard."""

from app.tabs.business import render_business_validation_tab
from app.tabs.inference import render_inference_tab
from app.tabs.training import render_training_tab

__all__ = [
    "render_training_tab",
    "render_business_validation_tab",
    "render_inference_tab",
]
