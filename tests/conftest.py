import numpy as np
import pandas as pd
import pytest

from src.config import ColumnMapping


@pytest.fixture
def transactions():
    rows = []
    for customer in range(1, 13):
        for order in range(3):
            rows.append(
                {
                    "customer": f"C{customer}",
                    "invoice": f"{customer}-{order}",
                    "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=customer * 4 + order * (customer % 3 + 1)),
                    "amount": float(customer * (order + 1)),
                    "category": f"category-{(customer + order) % 4}",
                    "promo": float(customer * (order + 1)) * 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def full_mapping():
    return ColumnMapping(
        customer_id="customer",
        order_id="invoice",
        purchase_date="date",
        spend="amount",
        category="category",
        promotional_spend="promo",
    )


@pytest.fixture
def clustered_points():
    rng = np.random.default_rng(17)
    return np.vstack(
        (
            rng.normal(loc=-3, scale=0.2, size=(12, 2)),
            rng.normal(loc=3, scale=0.2, size=(12, 2)),
        )
    )
