"""Dashboard generation: self-containment, palette discipline, honest framing."""

from __future__ import annotations

import re

import pytest

from swingbot.agents.baselines import AlwaysFlat, BuyAndHold, RandomAgent
from swingbot.backtest.runner import evaluate
from swingbot.config import EnvConfig, FeatureConfig, RiskConfig
from swingbot.dashboard import _MAX_SERIES, StrategyResult, build_dashboard
from swingbot.data.sources import SyntheticSource
from swingbot.features.technical import build_dataset, feature_columns


@pytest.fixture(scope="module")
def strategies() -> list[StrategyResult]:
    bars = SyntheticSource(seed=3).fetch("TEST", "2015-01-01", "2020-12-31")
    data = build_dataset(bars, FeatureConfig())
    cols = feature_columns(FeatureConfig())
    cfg = EnvConfig(episode_length=None, risk=RiskConfig(vol_target_annual=None))
    out = []
    for name, agent in [
        ("buy_and_hold", BuyAndHold()),
        ("flat", AlwaysFlat()),
        ("random", RandomAgent(seed=1)),
    ]:
        result, report = evaluate(data, cols, agent, cfg, n_trials=3)
        out.append(StrategyResult(name=name, result=result, report=report))
    return out


class TestBuildDashboard:
    def test_writes_a_file(self, strategies, tmp_path):
        p = build_dashboard(strategies, tmp_path / "d.html", symbol="TEST")
        assert p.exists() and p.stat().st_size > 5_000

    def test_is_fully_self_contained(self, strategies, tmp_path):
        """A strict-CSP / offline page must not reach for a CDN."""
        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        assert not re.search(r'src\s*=\s*["\']https?://', html)
        assert not re.search(r'href\s*=\s*["\']https?://', html)
        assert "cdn." not in html

    def test_declares_simulated_capital_prominently(self, strategies, tmp_path):
        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        assert "SIMULATED CAPITAL" in html
        assert "no order was ever placed" in html

    def test_ships_a_table_view_as_contrast_relief(self, strategies, tmp_path):
        """Three light-mode slots are sub-3:1, which obligates relief."""
        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        assert "<table class='metrics'>" in html
        assert "Deflated Sharpe" in html

    def test_defines_both_light_and_dark_steps(self, strategies, tmp_path):
        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        assert "prefers-color-scheme: dark" in html
        assert '[data-theme="dark"]' in html
        assert '[data-theme="light"]' in html  # toggle must win over OS dark

    def test_series_get_fixed_palette_slots_in_order(self, strategies, tmp_path):
        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        # Slot 1 blue, slot 2 green, slot 3 magenta -- never cycled.
        assert "#2a78d6" in html and "#008300" in html and "#e87ba4" in html

    def test_rejects_more_series_than_palette_slots(self, strategies, tmp_path):
        """A 6th hue would have to be invented; fold to 'Other' instead."""
        too_many = strategies * 3
        assert len(too_many) > _MAX_SERIES
        with pytest.raises(ValueError, match="exceeds"):
            build_dashboard(too_many, tmp_path / "d.html")

    def test_rejects_empty_input(self, tmp_path):
        with pytest.raises(ValueError, match="no strategies"):
            build_dashboard([], tmp_path / "d.html")

    def test_embedded_data_is_valid_json(self, strategies, tmp_path):
        import json

        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
        assert m, "embedded payload not found"
        payload = json.loads(m.group(1))
        assert len(payload["series"]) == len(strategies)
        assert all("equity" in s and "drawdown" in s for s in payload["series"])

    def test_downsamples_long_series_but_keeps_the_endpoint(self, strategies, tmp_path):
        import json

        html = build_dashboard(strategies, tmp_path / "d.html").read_text()
        payload = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
        s = payload["series"][0]
        assert len(s["equity"]) <= 901
        # Final equity must survive downsampling -- it is the direct label.
        assert s["equity"][-1] == pytest.approx(float(strategies[0].result.equity[-1]), rel=1e-4)
        assert len(s["x"]) == len(s["equity"])


def test_verdict_names_the_benchmark_honestly(strategies, tmp_path):
    """The headline must not bury a loss to buy-and-hold."""
    html = build_dashboard(strategies, tmp_path / "d.html").read_text()
    assert "Verdict:" in html
    assert "buy_and_hold" in html
    assert "consistent with having searched" in html
