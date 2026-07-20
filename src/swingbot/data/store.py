--- a/src/swingbot/data/store.py
+++ b/src/swingbot/data/store.py
@@
     def write(self, df: pl.DataFrame, *, validate_quality: bool = True) -> int:
         """Upsert bars. Existing (symbol, ts) rows are replaced by the new ones."""
         df = normalize(df)
         if validate_quality:
             problems = validate(df)
             if problems:
                 raise DataQualityError("; ".join(problems))
@@
-            if path.exists():
-                existing = pl.read_parquet(path)
-                group = normalize(pl.concat([existing, group], how="vertical"))
+            if path.exists():
+                existing = pl.read_parquet(path)
+                # Coerce incoming group to existing schema where possible to
+                # avoid Polars SchemaError on concat when dtypes differ. This
+                # can happen if earlier runs wrote integer-like columns while
+                # current in-memory values are Float64 (or vice versa).
+                for name, dtype in existing.schema.items():
+                    if name in group.columns:
+                        group = group.with_columns(pl.col(name).cast(dtype, strict=False))
+                group = normalize(pl.concat([existing, group], how="vertical"))
             group.write_parquet(path, compression="zstd")
             written += group.height
         return written
@@
     def read(
@@
-        df = pl.concat([pl.read_parquet(p) for p in existing], how="vertical")
+        df = pl.concat([pl.read_parquet(p) for p in existing], how="vertical")
         if start is not None:
             df = df.filter(pl.col("ts") >= _bound(start, df.schema["ts"]))
         if end is not None:
             df = df.filter(pl.col("ts") <= _bound(end, df.schema["ts"]))
 
         df = apply_adjustment(df, adjustment)
         return df.sort(["symbol", "ts"])
