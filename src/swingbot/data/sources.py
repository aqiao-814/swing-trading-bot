--- a/src/swingbot/data/sources.py
+++ b/src/swingbot/data/sources.py
@@
-        if not frames:
-            raise DataQualityError(f"no data fetched for any of {symbols}")
-        return normalize(pl.concat(frames, how="vertical"))
+        if not frames:
+            raise DataQualityError(f"no data fetched for any of {symbols}")
+
+        # Normalize any numeric columns that downstream code expects to be
+        # integer-like (notably 'invested' which may be produced later when
+        # writing ledger rows). Polars concatenation will raise a SchemaError
+        # if types differ across frames (e.g. some frames have Float64 while
+        # others have Int64). To avoid that, ensure column types are stable
+        # across frames before concat.
+        def _normalize_frame_types(df: pl.DataFrame) -> pl.DataFrame:
+            # If a frame carries an 'invested' column (rare for raw bars), make
+            # it Int64 after filling nulls. This is defensive: the ledger (not
+            # the bar frames) currently contains 'invested', but making this
+            # transformation here avoids surprises if intermediate frames
+            # are materialised with that column.
+            if "invested" in df.columns:
+                df = df.with_columns(pl.col("invested").fill_null(0).cast(pl.Int64))
+            return df
+
+        frames = [_normalize_frame_types(f) for f in frames]
+        return normalize(pl.concat(frames, how="vertical"))
