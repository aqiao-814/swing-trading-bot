--- a/src/swingbot/paper/engine.py
+++ b/src/swingbot/paper/engine.py
@@
-            if todo:
-                state.capture_portfolio(pf, entry_ts)
-                state.last_processed = todo[-1].isoformat()
-                ckpt_ts = todo[-1].date() if isinstance(todo[-1], datetime) else todo[-1]
-                self.store.append("ledger", pl.DataFrame(ledger_rows))
+            if todo:
+                state.capture_portfolio(pf, entry_ts)
+                state.last_processed = todo[-1].isoformat()
+                ckpt_ts = todo[-1].date() if isinstance(todo[-1], datetime) else todo[-1]
+                # Ensure ledger numeric columns have stable dtypes before append
+                ledger_df = pl.DataFrame(ledger_rows)
+                if "invested" in ledger_df.columns:
+                    ledger_df = ledger_df.with_columns(pl.col("invested").cast(pl.Float64))
+                if "cash" in ledger_df.columns:
+                    ledger_df = ledger_df.with_columns(pl.col("cash").cast(pl.Float64))
+                if "equity" in ledger_df.columns:
+                    ledger_df = ledger_df.with_columns(pl.col("equity").cast(pl.Float64))
+                if "daily_return" in ledger_df.columns:
+                    ledger_df = ledger_df.with_columns(pl.col("daily_return").cast(pl.Float64))
+                self.store.append("ledger", ledger_df)
@@
-            if trade_rows:
-                self.store.append("trades", pl.DataFrame(trade_rows))
+            if trade_rows:
+                self.store.append("trades", pl.DataFrame(trade_rows))
