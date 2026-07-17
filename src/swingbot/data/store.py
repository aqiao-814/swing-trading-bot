"""Bar storage: Parquet on disk, DuckDB as the query engine.

Chosen per the blueprint: columnar, compressed, no server, and DuckDB reads the
Parquet files directly so there is no import step and no second copy of the data.
Partitioned by symbol so a single-ticker read touches exactly one file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from swingbot.data.schema import (
    AdjustmentMode,
    DataQualityError,
    apply_adjustment,
    normalize,
    validate,
)


class BarStore:
    """Read/write daily bars, partitioned ``<root>/bars/symbol=<SYM>/bars.parquet``."""

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    # ---- paths -----------------------------------------------------------

    def _partition(self, symbol: str) -> Path:
        return self.bars_dir / f"symbol={symbol.upper()}" / "bars.parquet"

    def symbols(self) -> list[str]:
        return sorted(
            p.name.split("=", 1)[1]
            for p in self.bars_dir.glob("symbol=*")
            if (p / "bars.parquet").exists()
        )

    def __contains__(self, symbol: str) -> bool:
        return self._partition(symbol).exists()

    # ---- write -----------------------------------------------------------

    def write(self, df: pl.DataFrame, *, validate_quality: bool = True) -> int:
        """Upsert bars. Existing (symbol, ts) rows are replaced by the new ones."""
        df = normalize(df)
        if validate_quality:
            problems = validate(df)
            if problems:
                raise DataQualityError("; ".join(problems))

        written = 0
        for (symbol,), group in df.group_by(["symbol"], maintain_order=True):
            path = self._partition(str(symbol))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pl.read_parquet(path)
                group = normalize(pl.concat([existing, group], how="vertical"))
            group.write_parquet(path, compression="zstd")
            written += group.height
        return written

    # ---- read ------------------------------------------------------------

    def read(
        self,
        symbols: str | list[str] | None = None,
        *,
        start: str | date | None = None,
        end: str | date | None = None,
        adjustment: AdjustmentMode = AdjustmentMode.ADJUSTED,
    ) -> pl.DataFrame:
        """Read bars for symbols in [start, end], adjusted at read time."""
        if isinstance(symbols, str):
            symbols = [symbols]
        wanted = [s.upper() for s in symbols] if symbols else self.symbols()

        paths = [self._partition(s) for s in wanted]
        existing = [p for p in paths if p.exists()]
        if not existing:
            return pl.DataFrame(schema={"symbol": pl.Utf8, "ts": pl.Date})

        df = pl.concat([pl.read_parquet(p) for p in existing], how="vertical")
        if start is not None:
            df = df.filter(pl.col("ts") >= pl.lit(start).cast(pl.Date))
        if end is not None:
            df = df.filter(pl.col("ts") <= pl.lit(end).cast(pl.Date))

        df = apply_adjustment(df, adjustment)
        return df.sort(["symbol", "ts"])

    def sql(self, query: str) -> pl.DataFrame:
        """Run DuckDB SQL against the whole bar store.

        The hive-partitioned glob is exposed as the ``bars`` view, so callers can
        write ``SELECT * FROM bars WHERE symbol = 'AAPL'`` without knowing paths.
        """
        glob = str(self.bars_dir / "**" / "*.parquet")
        con = duckdb.connect()
        try:
            con.execute(
                f"CREATE VIEW bars AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1)"
            )
            return con.execute(query).pl()
        finally:
            con.close()

    # ---- introspection ---------------------------------------------------

    def coverage(self) -> pl.DataFrame:
        """Per-symbol first/last bar and row count -- the first thing to check."""
        frames = []
        for symbol in self.symbols():
            df = pl.read_parquet(self._partition(symbol))
            if df.is_empty():
                continue
            frames.append(
                pl.DataFrame(
                    {
                        "symbol": [symbol],
                        "bars": [df.height],
                        "start": [df["ts"].min()],
                        "end": [df["ts"].max()],
                    }
                )
            )
        if not frames:
            return pl.DataFrame(schema={"symbol": pl.Utf8, "bars": pl.Int64})
        return pl.concat(frames).sort("symbol")

    def trading_calendar(self, symbols: list[str] | None = None) -> list[date]:
        """Union of dates present across symbols -- the simulation's clock.

        Derived from the data rather than a holiday library, so the calendar can
        never disagree with the bars we actually have.
        """
        df = self.read(symbols, adjustment=AdjustmentMode.NONE)
        if df.is_empty():
            return []
        return df["ts"].unique().sort().to_list()
