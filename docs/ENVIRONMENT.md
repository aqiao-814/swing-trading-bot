# Environment notes (macOS / Apple Silicon)

Three environment problems were found on this machine while building the system.
All three fail **silently**, so check them before debugging anything else.

## 1. The default `python3` is Rosetta-emulated (breaks MPS)

The `python3` on PATH is pyenv 3.10.1 and reports:

```
platform.machine() -> 'x86_64'
```

That is an x86_64 binary running under Rosetta 2. Under emulation
`torch.backends.mps.is_available()` returns **False** and Lightning raises
`MPSAccelerator not available` — the GPU is simply invisible, with no error
explaining why. Most other 3.12 interpreters on this box are also x86_64
(Homebrew `/usr/local/bin/python3.12`, Anaconda `/opt/anaconda3`).

**This project pins the uv-managed arm64 build:**

```
/Users/aqiao/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
```

`.venv` is created from that interpreter, so `platform.machine() == 'arm64'` and
`torch.backends.mps.is_available() == True`.

Watch out: `uv run --python 3.12` may still resolve to an **x86_64** interpreter.
Pass the explicit path. Always verify:

```bash
./.venv/bin/python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"
# expect: arm64 True
```

## 2. iCloud silently breaks `uv pip install -e .`

**Symptom:** `import swingbot` works under pytest but raises `ModuleNotFoundError`
everywhere else, even though `uv pip install -e .` reported success.

**Cause:** a chain of three things.

1. The project used to live under `~/Desktop`, which is **iCloud-synced**
   (`brctl status` shows `com.apple.CloudDocs` actively syncing).
2. iCloud sets the macOS `UF_HIDDEN` flag on the `.pth` files in
   `.venv/lib/python3.12/site-packages/`.
3. CPython's `site.addpackage()` **skips any `.pth` file with `UF_HIDDEN` set** —
   no warning, no error. The editable install's path is never added to
   `sys.path`, so the install is a silent no-op.

`chflags nohidden` fixed it for about **25 seconds** before iCloud re-hid it
(measured, not guessed). It was not a durable fix.

**Resolution: the project was moved to `~/dev/swing-trading-bot`, outside iCloud.**
There the flag is never set and `uv pip install -e .` simply works — no
`PYTHONPATH` needed. **Do not move the project back under `~/Desktop`,
`~/Documents`, or any other iCloud-synced path.**

If you ever suspect this has returned:

```bash
ls -lO .venv/lib/python3.12/site-packages/*.pth   # "hidden" means you're in iCloud
```

The `Makefile` still exports `PYTHONPATH=src` and `pyproject.toml` still sets
`pythonpath = ["src"]`. Both are now belt-and-braces rather than load-bearing.

Note the `.venv` is **not relocatable** — absolute paths are baked into it. If you
move the project, delete `.venv` and recreate it (see section 1 for the correct
interpreter).

## 3. Why iCloud is a bad neighbour for this project generally

Beyond the `.pth` problem, an iCloud-synced project directory means:

- **`data/`** (Parquet market data, grows to GBs) is uploaded to iCloud.
- **`.venv/`** (thousands of files) is uploaded to iCloud. Here it was **1.2 GB**,
  versus 4.6 MB of actual source, data, and artifacts.
- With *Optimize Mac Storage* on, iCloud can **evict** files to the cloud and
  replace them with `.icloud` stubs — a long training run can fail mid-flight
  because its data was evicted.

Nothing in the code depends on the project's location, so staying outside iCloud
costs nothing.

## Compute notes

- **CPU is the right default** for these policies. RL trading nets are tiny
  (a few 64–256-unit layers) with small batches; they cannot fill a GPU, and the
  M4's AMX coprocessor makes small matmuls fast on CPU. MPS pays off only for
  transformer-scale components (e.g. a Decision Transformer).
- **float32 everywhere.** MPS cannot convert float64 tensors at all. The env's
  observation space is `float32` by construction for this reason.
- **Vectorized envs:** macOS spawns (not forks) subprocesses. Prefer
  `DummyVecEnv`; if you use `SubprocVecEnv`, guard under `if __name__ == "__main__":`.
