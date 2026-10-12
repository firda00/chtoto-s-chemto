from __future__ import annotations

from pathlib import Path

import pytest

from bank_auditor.categorize import categorize
from bank_auditor.parsers import load_statement
from bank_auditor.pipeline import build_report

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def sample_txns():
    return categorize(load_statement(DATA / "statement_sample.csv").transactions)


@pytest.fixture(scope="session")
def sample_report(sample_txns):
    return build_report(
        sample_txns, source_file="statement_sample.csv", format_name="csv/generic_en"
    )


@pytest.fixture(scope="session")
def subs_by_merchant(sample_report):
    return {s.merchant: s for s in sample_report.subscriptions}
