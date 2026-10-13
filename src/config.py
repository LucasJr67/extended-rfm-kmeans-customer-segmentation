"""Feature definitions and project-wide defaults."""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ColumnMapping:
    """Explicit mapping from source transaction columns to their meanings."""

    customer_id: str
    order_id: str
    purchase_date: str
    spend: str
    category: str | None = None
    promotional_spend: str | None = None


FEATURE_DEFINITIONS: Final[dict[str, str]] = {
    "R": "Analysis date minus the customer's most recent purchase date, in days.",
    "F": "Count of distinct order/invoice identifiers for the customer.",
    "M": "Sum of the mapped transaction-level spend values for the customer.",
    "D": "Count of distinct non-missing product categories for the customer.",
    "PS": "Sum of mapped promotional spending divided by total customer spending.",
    "PR": (
        "Sample standard deviation (ddof=1), in days, of gaps between consecutive "
        "distinct purchase dates; missing when fewer than three dates exist."
    ),
}

FEATURE_SETS: Final[dict[str, tuple[str, ...]]] = {
    "traditional_rfm": ("R", "F", "M"),
    "extended_rfm": ("R", "F", "M", "D", "PS", "PR"),
}

DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_N_INIT: Final[int] = 10
DEFAULT_K_VALUES: Final[tuple[int, ...]] = tuple(range(2, 11))
DEFAULT_METRIC_SAMPLE_LIMIT: Final[int] = 2000
