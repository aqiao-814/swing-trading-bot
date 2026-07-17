"""Continual RRL: one policy, many symbols, learning that never stops.

This wraps the existing ``RRLAgent`` (Moody & Saffell direct reinforcement)
rather than reimplementing it. The extension is twofold:

* **Cross-sectional weight sharing.** A single set of policy parameters
  ``(w, u, b)`` scores every symbol in the universe. Each symbol keeps its own
  recurrent state (``F_{t-1}`` and its gradient trace), which is swapped into
  the shared agent before any call. Every symbol's next-day outcome becomes a
  gradient step on the same weights, so one trading day of a 100-name universe
  is 100 experiences -- wins push the policy toward what worked, losses push it
  away, exactly as the DSR gradient dictates.

* **Persistence.** The full learner state (weights, per-symbol recurrences,
  DSR moments, update counters) serialises to a single ``.bin`` file, so
  learning survives restarts and each day's update is applied exactly once.

The reward each update sees is the *net* return of the position the policy
took: ``r = F_{t-1} * ret - cost * |F_t - F_{t-1}|`` -- costs are inside the
gradient, which is what stops continual learning from drifting into churn.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from swingbot.agents.baselines import RRLAgent

_FORMAT_VERSION = 2


@dataclass
class _SymbolState:
    """Per-symbol recurrent state for the shared policy."""

    f_prev: float = 0.0
    dfprev_dw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dfprev_du: float = 0.0


class ContinualRRL:
    """Shared-weight RRL policy over a universe, with online continual updates."""

    def __init__(
        self,
        feature_cols: list[str],
        *,
        learning_rate: float = 0.01,
        eta: float = 0.01,
        seed: int = 7,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.agent = RRLAgent(
            len(feature_cols), learning_rate=learning_rate, eta=eta, seed=seed, discrete=False
        )
        self._states: dict[str, _SymbolState] = {}
        self.n_updates = 0
        self.cum_reward = 0.0

    # ---- per-symbol state plumbing ----------------------------------------

    def _state(self, symbol: str) -> _SymbolState:
        st = self._states.get(symbol)
        if st is None:
            st = _SymbolState(dfprev_dw=np.zeros(self.agent.n_features))
            self._states[symbol] = st
        return st

    def _load_state(self, st: _SymbolState) -> None:
        self.agent.f_prev = st.f_prev
        self.agent._dfprev_dw = st.dfprev_dw
        self.agent._dfprev_du = st.dfprev_du

    def _save_state(self, st: _SymbolState) -> None:
        st.f_prev = self.agent.f_prev
        st.dfprev_dw = self.agent._dfprev_dw
        st.dfprev_du = self.agent._dfprev_du

    # ---- inference ---------------------------------------------------------

    def score(self, symbol: str, x: np.ndarray) -> float:
        """Conviction in [-1, 1] for one symbol. Pure: mutates no state.

        Positive = long conviction, negative = short. Magnitude is how sure the
        policy is; it feeds both ranking and position sizing.
        """
        st = self._state(symbol)
        saved = self.agent.f_prev
        self.agent.f_prev = st.f_prev
        f = self.agent.position(np.asarray(x, dtype=np.float64))
        self.agent.f_prev = saved
        return f

    # ---- learning ----------------------------------------------------------

    def observe(self, symbol: str, x: np.ndarray, ret: float, cost: float) -> float:
        """One continual-learning step for one symbol.

        ``x`` is the feature vector the day's decision was made on and ``ret``
        the next completed bar's return -- so every call is a genuinely
        realized experience, never a forecast. Returns the net reward ``r``.
        """
        st = self._state(symbol)
        self._load_state(st)
        r = self.agent.update(np.asarray(x, dtype=np.float64), float(ret), float(cost))
        self._save_state(st)
        self.n_updates += 1
        self.cum_reward += r
        return r

    def pretrain(
        self,
        features_by_symbol: dict[str, np.ndarray],
        returns_by_symbol: dict[str, np.ndarray],
        *,
        cost: float,
        epochs: int = 1,
    ) -> None:
        """Warm-start on history so day one is informed rather than random.

        Runs the same ``observe`` path over each symbol's past bars, in sorted
        symbol order for determinism. Pretraining counts as updates -- the
        learner does not distinguish rehearsed experience from live experience.
        """
        for _ in range(max(epochs, 0)):
            for symbol in sorted(features_by_symbol):
                x, rets = features_by_symbol[symbol], returns_by_symbol[symbol]
                for t in range(min(len(x), len(rets))):
                    self.observe(symbol, x[t], float(rets[t]), cost)

    # ---- diagnostics --------------------------------------------------------

    @property
    def sharpe(self) -> float:
        """The policy's own EW Sharpe estimate over everything it has lived."""
        return self.agent.sharpe

    @property
    def avg_reward(self) -> float:
        return self.cum_reward / self.n_updates if self.n_updates else 0.0

    def weight_norm(self) -> float:
        return float(np.linalg.norm(self.agent.w))

    # ---- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise the complete learner to one file (npz payload)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        symbols = sorted(self._states)
        n = self.agent.n_features
        buf = io.BytesIO()
        np.savez(
            buf,
            version=np.int64(_FORMAT_VERSION),
            feature_cols=np.array(self.feature_cols),
            w=self.agent.w,
            u=np.float64(self.agent.u),
            b=np.float64(self.agent.b),
            lr=np.float64(self.agent.lr),
            eta=np.float64(self.agent.eta),
            a=np.float64(self.agent.a),
            b_moment=np.float64(self.agent.b_moment),
            steps=np.int64(self.agent._steps),
            n_updates=np.int64(self.n_updates),
            cum_reward=np.float64(self.cum_reward),
            symbols=np.array(symbols),
            f_prev=np.array([self._states[s].f_prev for s in symbols]),
            dfprev_dw=(
                np.stack([self._states[s].dfprev_dw for s in symbols])
                if symbols
                else np.zeros((0, n))
            ),
            dfprev_du=np.array([self._states[s].dfprev_du for s in symbols]),
        )
        path.write_bytes(buf.getvalue())
        return path

    @classmethod
    def load(cls, path: str | Path) -> ContinualRRL:
        with np.load(io.BytesIO(Path(path).read_bytes()), allow_pickle=False) as z:
            feature_cols = [str(c) for c in z["feature_cols"]]
            learner = cls(feature_cols, learning_rate=float(z["lr"]), eta=float(z["eta"]))
            learner.agent.w = z["w"].astype(np.float64)
            learner.agent.u = float(z["u"])
            learner.agent.b = float(z["b"])
            learner.agent.a = float(z["a"])
            learner.agent.b_moment = float(z["b_moment"])
            learner.agent._steps = int(z["steps"])
            learner.n_updates = int(z["n_updates"])
            learner.cum_reward = float(z["cum_reward"])
            for i, symbol in enumerate(str(s) for s in z["symbols"]):
                learner._states[symbol] = _SymbolState(
                    f_prev=float(z["f_prev"][i]),
                    dfprev_dw=z["dfprev_dw"][i].astype(np.float64),
                    dfprev_du=float(z["dfprev_du"][i]),
                )
        return learner

    def checkpoint(self, models_root: str | Path, ts: date, *, max_keep: int = 30) -> Path:
        """Write ``rrl_latest.bin`` plus a dated checkpoint, pruning old ones."""
        root = Path(models_root)
        latest = self.save(root / "rrl_latest.bin")
        ckpt_dir = root / "checkpoints"
        self.save(ckpt_dir / f"rrl_{ts.isoformat()}.bin")
        checkpoints = sorted(ckpt_dir.glob("rrl_*.bin"))
        for old in checkpoints[: max(len(checkpoints) - max_keep, 0)]:
            old.unlink()
        manifest = {
            "latest": latest.name,
            "updated": ts.isoformat(),
            "n_updates": self.n_updates,
            "checkpoints": [p.name for p in sorted(ckpt_dir.glob("rrl_*.bin"))],
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return latest
