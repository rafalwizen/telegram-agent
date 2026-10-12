"""Tests for GLM client helpers (quota detection)."""

from __future__ import annotations

from tgagent.glm_client import is_quota_error


def test_balance_error_is_quota():
    assert is_quota_error(
        "Error code: 429 - {'error': {'code': '1113',"
        " 'message': 'Insufficient balance or no resource package. Please recharge.'}}"
    )


def test_window_limit_is_quota():
    assert is_quota_error("Usage limit reached for the current 5-hour window")
    assert is_quota_error("prompt quota exceeded for this account")


def test_transient_errors_are_not_quota():
    assert not is_quota_error("Error code: 429 - too many requests, slow down")
    assert not is_quota_error("Error code: 500 - internal server error")
    assert not is_quota_error("connection reset by peer")
