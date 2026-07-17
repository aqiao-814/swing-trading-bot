# Continual RRL: how the agent learns from real forward performance

The paper-trading agent is Moody & Saffell's Recurrent Reinforcement Learning
policy (*Learning to Trade via Direct Reinforcement*, IEEE TNN 2001) — the same
`RRLAgent` used by the backtester — extended into a **continually learning,
cross-sectional** policy by `swingbot.paper.learner.ContinualRRL`.

## The policy

One shared parameter set scores every stock:

```
F_t(symbol) = tanh( w · x_t(symbol) + u · F_{t-1}(symbol) + b )
```

- `w, u, b` are **shared across the whole universe** — a few dozen parameters
  in total, which on low-signal daily data is a feature, not a limitation.
- Each symbol keeps its **own recurrent state** `F_{t-1}` (and its gradient
  trace), swapped into the shared agent before any call. The recurrent term is
  what makes the policy cost-aware: its own prior position enters the decision,
  so it can learn that holding is cheaper than flipping.

The output `F ∈ [-1, 1]` is the **conviction score**: sign is direction,
magnitude is sizing. It drives ranking, allocation
(`weight = F × max_position_weight`), and exits (conviction decay).

## What "experience" is

Every completed trading day, every feature-complete symbol contributes one
update. The reward for a symbol's update is the **net** return of the position
the policy held:

```
r = F_{t-1} · ret_t  −  cost · |F_t − F_{t-1}|
```

where `ret_t` is the realized close-to-close return of the newest completed bar
and `cost` is the round-trip friction estimate (spread + slippage). The
gradient ascends the **differential Sharpe ratio** — an exponentially-weighted
online derivative of the Sharpe ratio — so:

- winning positions *reinforce* the weights that produced them,
- losing positions push the policy away from repeating them,
- volatile wins score worse than steady ones (variance enters the DSR),
- and churn is penalized inside the gradient itself, not by an external rule.

A 100-name universe therefore generates ~100 real experiences per day, all
flowing into the same weights. Updates happen once per (symbol, day) — the
engine's idempotency watermark guarantees a day can never be trained on twice.

## Timing

At the close of day *d* the learner is updated with the feature vector from
day *d−1* and the return *d−1 → d* — the same one-bar execution delay the
simulator enforces, so the policy is never credited with a return it could not
have captured.

## Warm start, then forward-only

On first run the policy **pretrains** on `paper.pretrain_years` of history
strictly *before* inception (same update rule, replayed over the past), so day
one is informed rather than a random coin-flip. Everything after inception is
learned exclusively from forward, unseen market data — which is the point:
backtests can be fitted; a forward track record cannot.

## Persistence

The complete learner — weights, per-symbol recurrent states, DSR moments,
update counters — serialises to `artifacts/models/rrl_latest.bin` (an npz
payload), with dated copies under `artifacts/models/checkpoints/` pruned to
`paper.max_checkpoints`. Learning survives restarts bit-for-bit (tested), and
the saved file records its feature columns: loading a model against a config
with different features fails loudly instead of silently misreading weights.

## Learning metrics

`artifacts/paper/portfolio/learning.parquet` tracks per day: total training
iterations, updates that day, mean day reward, cumulative reward, policy loss
(the negated mean DSR reward — this is direct policy gradient ascent, there is
no value function), the policy's own EW Sharpe estimate, and the weight norm.
The dashboard's *Learning progress* section plots cumulative reward and EW
Sharpe. The policy is deterministic, so there is no exploration-rate schedule;
exploration comes from the breadth of the universe itself.

## Honest expectations

RRL after costs historically struggles to beat buy-and-hold (see the README's
backtest table — that result is the finding, not a bug). Continual learning
does not repeal that; it measures it on data nobody could have fitted to. Judge
the agent by the dashboard's benchmark table over months, not by any single
day's P&L.
