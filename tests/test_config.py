"""Tests for the configuration layer.

The real config.json is valid, so these tests exercise `_load_defaults`
(a pure function) with synthetic files to prove the fail-fast behavior
(F-12) without touching the repo's actual configuration or module state.
"""
from __future__ import annotations

import json

import pytest

from src.config import _load_defaults


def test_corrupt_config_fails_fast(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(SystemExit, match="corrupt or unreadable"):
        _load_defaults(bad)


def test_non_dict_config_fails_fast(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(SystemExit, match="must be a JSON object"):
        _load_defaults(bad)


def test_missing_config_returns_defaults(tmp_path):
    assert _load_defaults(tmp_path / "nope.json") == {}


def test_valid_config_loads(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps({"max_upload_size_mb": 7}), encoding="utf-8")
    assert _load_defaults(good) == {"max_upload_size_mb": 7}


def test_invalid_tainted_policy_fails():
    """Field validators reject bogus enum-ish values (config correctness)."""
    from pydantic import ValidationError
    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(tainted_policy="bogus", api_auth_token="x")


def test_warn_if_server_keyed_without_auth(caplog):
    """F2-15: the free-LLM-proxy warning fires exactly in the danger scenario."""
    import logging
    import src.config as config_module

    with caplog.at_level(logging.WARNING, logger="src.config"):
        mock_self = type("MockSelf", (), {})()
        mock_self.openrouter_api_key = "sk-or-server-key"
        mock_self.api_auth_token = ""
        config_module.Settings.warn_if_server_keyed_without_auth(mock_self)

        assert any(
            "api_auth_token" in r.message and r.name == "src.config" for r in caplog.records
        ), "expected a warning when a server-side key is set without auth"


def test_no_warning_with_auth(caplog):
    import logging
    import src.config as config_module

    with caplog.at_level(logging.WARNING, logger="src.config"):
        mock_self = type("MockSelf", (), {})()
        mock_self.openrouter_api_key = "sk-or-server-key"
        mock_self.api_auth_token = "secret"
        config_module.Settings.warn_if_server_keyed_without_auth(mock_self)
        assert not any(r.name == "src.config" for r in caplog.records)