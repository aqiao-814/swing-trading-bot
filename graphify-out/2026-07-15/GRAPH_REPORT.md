# Graph Report - swing-trading-bot  (2026-07-15)

## Corpus Check
- 33 files · ~35,671 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 565 nodes · 1016 edges · 50 communities (20 shown, 30 thin omitted)
- Extraction: 67% EXTRACTED · 33% INFERRED · 0% AMBIGUOUS · INFERRED: 336 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- FeatureConfig
- Portfolio
- BarStore
- ValueError
- CostConfig
- DifferentialSharpeRatio
- evaluate
- metrics.py
- Agent
- frac_diff_ffd
- swingbot
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- FeatureConfig
- swingbot
- DataFrame
- EpisodeResult
- ndarray
- SwingTradingEnv
- ndarray
- Path
- StrEnum
- DataFrame
- StrEnum
- ABC
- DataFrame
- date
- Path
- DataFrame
- date
- Path
- ndarray
- DataFrame
- ndarray
- date
- DataFrame
- EpisodeResult
- Path
- ABC
- DataFrame
- SwingTradingEnv
- DataFrame

## God Nodes (most connected - your core abstractions)
1. `CostConfig` - 34 edges
2. `ExecutionModel` - 30 edges
3. `FeatureConfig` - 29 edges
4. `Portfolio` - 28 edges
5. `SyntheticSource` - 25 edges
6. `make_env()` - 22 edges
7. `build_report()` - 20 edges
8. `DifferentialSharpeRatio` - 20 edges
9. `RiskConfig` - 19 edges
10. `build_dashboard()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `bars()` --calls--> `SyntheticSource`  [INFERRED]
  tests/test_features.py → src/swingbot/data/sources.py
- `strategies()` --calls--> `BuyAndHold`  [INFERRED]
  tests/test_dashboard.py → src/swingbot/agents/baselines.py
- `TestBuildDashboard` --uses--> `BuyAndHold`  [INFERRED]
  tests/test_dashboard.py → src/swingbot/agents/baselines.py
- `strategies()` --calls--> `AlwaysFlat`  [INFERRED]
  tests/test_dashboard.py → src/swingbot/agents/baselines.py
- `TestBuildDashboard` --uses--> `AlwaysFlat`  [INFERRED]
  tests/test_dashboard.py → src/swingbot/agents/baselines.py

## Import Cycles
- None detected.

## Communities (50 total, 30 thin omitted)

### Community 0 - "FeatureConfig"
Cohesion: 0.08
Nodes (24): BaseModel, ActionSpaceKind, BacktestConfig, Config, DataConfig, EnvConfig, Typed configuration for the whole system.  Every knob that affects a backtest li, Where price data comes from and lives. (+16 more)

### Community 1 - "Portfolio"
Cohesion: 0.06
Nodes (20): Portfolio, PortfolioSnapshot, Position, Portfolio accounting.  This module is deliberately dumb about strategy and smart, Immutable record of portfolio state at one point in time., Cash + positions with broker-grade accounting.      The invariant that matters:, Total economic cost of trading: explicit charges plus slippage., Total account value: cash plus signed mark-to-market of holdings. (+12 more)

### Community 2 - "BarStore"
Cohesion: 0.09
Nodes (22): CostConfig, Trading frictions. Defaults model a retail zero-commission US equity broker., ExecutionModel, MarketContext, Execution and friction modelling.  Turns an *intent* ("I want to be 60% long AAP, Overnight borrow on short notional, accrued on a 360-day basis., Produce a fully-costed Fill, or None when there is nothing to trade., Estimated cost of entering and exiting, in bps of notional.          Useful as a (+14 more)

### Community 3 - "ValueError"
Cohesion: 0.09
Nodes (13): CombinatorialPurgedCV, PurgedKFold, Purged cross-validation (Lopez de Prado, AFML Ch. 7).  Standard k-fold leaks on, K-fold over time with purging and an embargo.      Parameters     ----------, Combinatorial Purged Cross-Validation (CPCV).      Partitions the series into ``, Split, Purging, embargo, and the overfitting statistics., The reason CPCV exists: many paths, hence many Sharpes. (+5 more)

### Community 4 - "CostConfig"
Cohesion: 0.07
Nodes (39): probability_of_backtest_overfitting(), PBO: P(the config that ranked best in-sample underperforms the median OOS)., annualized_volatility(), build_report(), cagr(), calmar_ratio(), conditional_value_at_risk(), deflated_sharpe_ratio() (+31 more)

### Community 5 - "DifferentialSharpeRatio"
Cohesion: 0.07
Nodes (29): AdjustmentMode, apply_adjustment(), DataQualityError, normalize(), Canonical bar schema and validation.  Every source normalises into this shape be, Apply split/dividend adjustment at read time.      ADJUSTED scales OHLC by ``adj, Raised when bars violate an invariant that would corrupt a backtest., Coerce a source frame into BAR_SCHEMA order, types, and sort. (+21 more)

### Community 6 - "evaluate"
Cohesion: 0.08
Nodes (21): RewardKind, build_reward(), DifferentialSharpeRatio, DrawdownPenalized, NetReturn, Reward functions.  Reward design is the make-or-break decision in trading RL. Ra, Plain net return, optionally log-scaled. The honest naive baseline.      Costs a, Net return minus a penalty proportional to current drawdown depth.      Targets (+13 more)

### Community 7 - "metrics.py"
Cohesion: 0.08
Nodes (17): Agent, AlwaysFlat, BuyAndHold, MovingAverageCrossover, RandomAgent, Baseline strategies -- the bar deep RL has to clear.  The blueprint is blunt abo, Take only the market features, dropping any appended agent state., Raw continuous position in [-1, 1]. (+9 more)

### Community 8 - "Agent"
Cohesion: 0.11
Nodes (18): build_dashboard(), _downsample(), _fmt(), _metrics_table(), ndarray, Path, Self-contained HTML analytics dashboard.  Renders equity curves, underwater (dra, Write a self-contained HTML dashboard. Returns the path. (+10 more)

### Community 9 - "frac_diff_ffd"
Cohesion: 0.12
Nodes (13): ffd_weights(), frac_diff_ffd(), min_ffd_order(), Fractional differentiation (Lopez de Prado, AFML Ch. 5).  The dilemma: raw price, Binomial weights for ``(1-L)^d``, truncated where |w| < threshold.      Recurren, Fixed-width fractionally-differenced series.      Returns an array the same leng, Smallest ``d`` whose FFD series passes an ADF stationarity test.      Returns ``, bars() (+5 more)

### Community 11 - "__init__.py"
Cohesion: 0.15
Nodes (12): Architecture, Code intelligence (graphify), Data, Known limitation: survivorship bias, Non-goals, Quick start, Realism modelled, Status (+4 more)

### Community 12 - "__init__.py"
Cohesion: 0.24
Nodes (10): drawdown_series(), equity_frame(), format_report(), _git_revision(), Artifact writing: trade logs, equity curves, run manifests.  Every run writes a, Full decision log: every bar, not just the ones that traded.      Keeping non-tr, Persist a complete, self-describing run directory., Human-readable metric block for the terminal. (+2 more)

### Community 13 - "__init__.py"
Cohesion: 0.22
Nodes (8): created_utc, git_revision, label, platform, python, seed, simulated_capital, starting_capital

### Community 16 - "FeatureConfig"
Cohesion: 0.06
Nodes (50): Expr, backtest(), _build_agent(), compare(), dashboard(), fetch(), _load_config(), DataFrame (+42 more)

### Community 17 - "swingbot"
Cohesion: 0.33
Nodes (5): 1. The default `python3` is Rosetta-emulated (breaks MPS), 2. iCloud silently breaks `uv pip install -e .`, 3. Why iCloud is a bad neighbour for this project generally, Compute notes, Environment notes (macOS / Apple Silicon)

## Knowledge Gaps
- **22 isolated node(s):** `label`, `created_utc`, `git_revision`, `python`, `platform` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CostConfig` connect `BarStore` to `FeatureConfig`, `FeatureConfig`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `SyntheticSource` connect `FeatureConfig` to `Agent`, `FeatureConfig`, `DifferentialSharpeRatio`, `frac_diff_ffd`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `evaluate()` connect `FeatureConfig` to `CostConfig`, `metrics.py`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `CostConfig` (e.g. with `ExecutionModel` and `MarketContext`) actually correct?**
  _`CostConfig` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `ExecutionModel` (e.g. with `CostConfig` and `Fill`) actually correct?**
  _`ExecutionModel` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `FeatureConfig` (e.g. with `strategies()` and `TestBuildDashboard`) actually correct?**
  _`FeatureConfig` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `Portfolio` (e.g. with `TestPortfolio` and `.test_borrow_charge_reduces_equity()`) actually correct?**
  _`Portfolio` has 12 INFERRED edges - model-reasoned connections that need verification._