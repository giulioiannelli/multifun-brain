#!/usr/bin/env python3
"""multifun-brain dashboard — one-command, cross-platform setup & launch.

Runs on Windows, macOS and Linux with **only Python already installed** (the
conda env in ``multifun-brain.yml`` provides that plus Node in one step — see
``SETUP.md``). From a fresh clone with the shared ``data/`` folder in place:

    python quickstart.py                 # set everything up, then open the dashboard

Everyday use after a ``git pull`` (deps + build already done, bundles cached):

    python quickstart.py serve           # just launch + open the browser

Other modes / knobs::

    python quickstart.py --dry-run       # show the plan (detected data, what
                                         # would be installed / ingested / built)
    python quickstart.py --data PATH     # use a data folder that lives elsewhere
                                         # (not copied — the server is pointed at it)
    python quickstart.py --no-ingest     # skip computing missing result bundles
    python quickstart.py --rebuild       # force a fresh frontend build
    python quickstart.py --no-serve      # set up only, don't launch
    python quickstart.py --port 8001 --host 0.0.0.0   # serving knobs

What it does, in order:
  1. Check the Python version.
  2. Install the package + dashboard deps into *this* interpreter (skipped when
     they already import — so repeat runs are instant).
  3. Inventory ``data/``: for every ``raw_data_<id>`` folder, detect its atlas,
     count subjects + processing variants, and check whether its result bundles
     (global / bands / patients) already exist.
  4. Ingest (compute + cache) the bundles that are missing. This is the only
     slow step and only runs the first time / for new datasets — it is skipped
     entirely when every dataset already has bundles.
  5. Build the frontend (only when there's no existing build, or ``--rebuild``).
  6. Serve everything from one local port and open the browser.

This script is pure standard-library on purpose: it must run *before* the
project's dependencies are installed. Heavier work is delegated to the library
(``scripts.raw_ingest``) and the frontend toolchain (``npm``).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── pretty output ──────────────────────────────────────────────────────────
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_C = {
    "step": "\033[1;36m",  # bold cyan
    "ok": "\033[1;32m",  # bold green
    "warn": "\033[1;33m",  # bold yellow
    "err": "\033[1;31m",  # bold red
    "dim": "\033[2m",
    "end": "\033[0m",
}


def _say(kind: str, msg: str) -> None:
    tag = {"step": "▶", "ok": "✓", "warn": "!", "err": "✗", "dim": " "}[kind]
    if _COLOR:
        print(f"{_C[kind]}{tag}{_C['end']} {msg}")
    else:
        print(f"{tag} {msg}")


def step(m: str) -> None:
    print()
    _say("step", m)


def ok(m: str) -> None:
    _say("ok", m)


def warn(m: str) -> None:
    _say("warn", m)


def die(m: str) -> None:
    _say("err", m)
    sys.exit(1)


# ── small cross-platform helpers ───────────────────────────────────────────
def have(module: str) -> bool:
    """True if *module* is importable in this interpreter (no import executed)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> int:
    """Echo and run a command; return its exit code (never raises on failure)."""
    printable = " ".join(str(c) for c in cmd)
    print(f"  {_C['dim'] if _COLOR else ''}$ {printable}{_C['end'] if _COLOR else ''}")
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env).returncode
    except FileNotFoundError:
        return 127


def npm_path() -> str | None:
    """Locate npm cross-platform (``npm.cmd`` on Windows via PATHEXT)."""
    return shutil.which("npm")


def run_npm(args: list[str], cwd: Path) -> int:
    """Run npm, tolerating Windows' ``.cmd`` shim quirks."""
    npm = npm_path()
    if not npm:
        return 127
    rc = run([npm, *args], cwd=cwd)
    if rc == 127 and os.name == "nt":  # last resort: let the shell resolve npm.cmd
        rc = subprocess.run(f"npm {' '.join(args)}", cwd=str(cwd), shell=True).returncode
    return rc


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((("127.0.0.1" if host in ("0.0.0.0", "") else host), port))
            return True
        except OSError:
            return False


# ── data folder resolution + inventory ─────────────────────────────────────
_ATLAS_BY_ID = (
    ("schaefer", "Schaefer-100"),
    ("harvardoxford", "HarvardOxford-48"),
    ("harvard_oxford", "HarvardOxford-48"),
)


def atlas_of(dataset_id: str) -> str:
    low = dataset_id.lower()
    for key, name in _ATLAS_BY_ID:
        if key in low:
            return name
    return "unknown"


def data_env(data_root: Path) -> dict[str, str]:
    """Environment overrides that point every backend root at *data_root*.

    Lets a collaborator keep the shared folder wherever it already lives
    (``--data PATH``) instead of copying it under the repo. Mirrors the defaults
    in ``dashboard/backend/config.py``; only keys whose target exists are set, so
    absent optional assets fall back to the backend's own defaults.
    """
    env: dict[str, str] = {"MFB_DATA_ROOT": str(data_root)}
    candidates = {
        "MFB_RESULTS_ROOT": data_root / "correlation_matrices_results",
        "MFB_ATLAS_DIR": data_root / "schaefer_2018",
        "MFB_HARVARD_OXFORD_XML": data_root / "HarvardOxford-Cortical.xml",
    }
    # Default raw set: prefer the April Schaefer set, else the first raw_data_*.
    raw_dirs = sorted(data_root.glob("raw_data_*"))
    preferred = data_root / "raw_data_schaefer100_april2026"
    default_raw = preferred if preferred.is_dir() else (raw_dirs[0] if raw_dirs else None)
    if default_raw:
        candidates["MFB_RAW_DATA_ROOT"] = default_raw
    for key, path in candidates.items():
        if path.exists():
            env[key] = str(path)
    return env


def inventory(data_root: Path, results_root: Path) -> list[dict]:
    """Detect every ``raw_data_<id>`` dataset and its computed-bundle status.

    Pure (no printing / side effects) so it can be unit-tested. Each entry::

        {id, atlas, n_subjects, n_variants|None, bundles: {global,bands,patients},
         has_bundles: bool, needs_ingest: bool}

    ``n_variants`` is ``None`` when the ``multifunbrain`` library isn't importable
    yet (pre-install) — subject counts still come from the folder structure.
    """
    discover = None
    if have("multifunbrain"):
        try:
            from multifunbrain.io import discover_timecourses as discover  # noqa: F811
        except Exception:  # noqa: BLE001 - inventory must never hard-fail
            discover = None

    out: list[dict] = []
    for raw in sorted(data_root.glob("raw_data_*")):
        if not raw.is_dir():
            continue
        ds_id = raw.name[len("raw_data_") :]
        n_subjects = sum(1 for p in raw.glob("sub-*") if p.is_dir())
        n_variants = None
        if discover is not None:
            try:
                n_variants = len({f.processing for f in discover(raw)})
            except Exception:  # noqa: BLE001
                n_variants = None
        bundles = {
            level: (results_root / ds_id / level / "results.pkl").exists()
            for level in ("global", "bands", "patients")
        }
        has_bundles = any(bundles.values())
        out.append(
            {
                "id": ds_id,
                "raw": raw,
                "atlas": atlas_of(ds_id),
                "n_subjects": n_subjects,
                "n_variants": n_variants,
                "bundles": bundles,
                "has_bundles": has_bundles,
                "needs_ingest": not all(bundles.values()),
            }
        )
    return out


def print_inventory(inv: list[dict], results_root: Path) -> None:
    if not inv:
        warn("no raw_data_* datasets found under the data folder.")
        return
    print(f"  {'dataset':<28} {'atlas':<16} {'subj':>4} {'vars':>4}  bundles")
    print(f"  {'-' * 28} {'-' * 16} {'-' * 4} {'-' * 4}  {'-' * 24}")
    for d in inv:
        got = [lvl for lvl, present in d["bundles"].items() if present]
        status = "·".join(got) if got else "— none (will compute)"
        v = "?" if d["n_variants"] is None else d["n_variants"]
        print(
            f"  {d['id']:<28} {d['atlas']:<16} {d['n_subjects']:>4} {str(v):>4}  {status}"
        )


# ── the pipeline stages ────────────────────────────────────────────────────
def ensure_python() -> None:
    if sys.version_info < (3, 9):  # noqa: UP036 - defensive: script runs pre-install on any Python
        die(f"Python ≥ 3.9 required; this is {sys.version.split()[0]}.")
    ok(f"Python {sys.version.split()[0]} ({sys.executable})")


def ensure_deps(args: argparse.Namespace) -> None:
    step("Checking Python dependencies")
    needed = ("multifunbrain", "fastapi", "uvicorn", "emd")
    missing = [m for m in needed if not have(m)]
    if not missing and not args.reinstall:
        ok("package + dashboard deps already installed.")
        return
    if args.dry_run:
        warn(f"would run: pip install -e '.[dashboard]'  (missing: {', '.join(missing) or 'reinstall'})")
        return
    warn(f"installing project + dashboard deps (missing: {', '.join(missing) or 'reinstall'})…")
    rc = run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dashboard]"],
        cwd=REPO_ROOT,
    )
    if rc != 0:
        die(
            "dependency install failed. If you use conda, the one-step env is:\n"
            "    conda env create -f multifun-brain.yml && conda activate multifun-brain\n"
            "then re-run:  python quickstart.py"
        )
    importlib.invalidate_caches()  # let a just-installed editable package import in-process
    ok("dependencies installed.")


def ensure_bundles(
    inv: list[dict], results_root: Path, env: dict[str, str], args: argparse.Namespace
) -> None:
    step("Result bundles (the data the result tabs read)")
    todo = [d for d in inv if d["needs_ingest"]]
    if not inv:
        warn("no raw datasets to compute from — the result tabs will be empty.")
        warn("place the shared data folder at ./data (see SETUP.md), or pass --data PATH.")
        return
    print_inventory(inv, results_root)
    if not todo:
        ok("every dataset already has bundles — nothing to compute.")
        return
    if args.no_ingest:
        warn(f"{len(todo)} dataset(s) lack bundles; --no-ingest set, skipping compute.")
        warn("their result tabs stay empty until you run the ingest.")
        return
    print()
    warn(
        f"{len(todo)} dataset(s) need computing: {', '.join(d['id'] for d in todo)}.\n"
        "    This is the one slow step (EMD + LRG over every matrix): minutes to a\n"
        "    few hours per dataset, ONCE. Results are cached; later runs skip it."
    )
    # Pass explicit raw + output paths (not just --dataset) so the ingest lands in
    # the same place the inventory looked — correct for ./data AND --data /elsewhere.
    def ingest_cmd(d: dict) -> list[str]:
        return [
            sys.executable, "-m", "scripts.raw_ingest.run_dataset",
            "--raw-root", str(d["raw"]),
            "--out", str(results_root / d["id"]),
        ]

    if args.dry_run:
        for d in todo:
            warn("would ingest: " + " ".join(ingest_cmd(d)))
        return
    child_env = {**os.environ, **env}
    for d in todo:
        step(f"Ingesting {d['id']} ({d['n_subjects']} subjects) — this can take a while")
        rc = run(ingest_cmd(d), cwd=REPO_ROOT, env=child_env)
        if rc != 0:
            warn(f"ingest of {d['id']} failed (rc={rc}); continuing with the rest.")
        else:
            ok(f"{d['id']} bundles cached under {results_root / d['id']}")


def ensure_frontend(args: argparse.Namespace) -> bool:
    """Build the React bundle if absent/forced. Returns True if a build exists."""
    step("Frontend (the dashboard UI)")
    fe = REPO_ROOT / "dashboard" / "frontend"
    dist_index = fe / "dist" / "index.html"
    if dist_index.exists() and not args.rebuild:
        ok("using existing frontend build (pass --rebuild to force a fresh one).")
        return True
    npm = npm_path()
    if not npm:
        if dist_index.exists():
            warn("Node/npm not found, but a build already exists — serving that.")
            return True
        warn(
            "Node.js/npm not found and no existing build. The UI can't be built.\n"
            "    Easiest fix (bundles Node cross-platform):\n"
            "        conda env create -f multifun-brain.yml && conda activate multifun-brain\n"
            "    or install Node ≥ 18 (https://nodejs.org) and re-run."
        )
        return False
    if args.dry_run:
        if not (fe / "node_modules").exists():
            warn("would run: npm install")
        warn("would run: npm run build")
        return True
    if not (fe / "node_modules").exists():
        warn("installing frontend dependencies (first run only)…")
        if run_npm(["install"], cwd=fe) != 0:
            die("npm install failed.")
    warn("building the frontend…")
    if run_npm(["run", "build"], cwd=fe) != 0:
        die("frontend build failed.")
    ok("frontend built.")
    return True


def serve(env: dict[str, str], args: argparse.Namespace) -> None:
    step("Launching the dashboard")
    url = f"http://localhost:{args.port}"
    if args.dry_run:
        note = "" if port_free(args.host, args.port) else "   (note: that port is currently busy)"
        warn(f"would launch: uvicorn dashboard.backend.app:app --host {args.host} --port {args.port}{note}")
        warn(f"would open:   {url}")
        return
    if not port_free(args.host, args.port):
        die(
            f"port {args.port} is already in use — a previous server may still be running.\n"
            f"    free it, or pick another:  python quickstart.py serve --port 8001"
        )
    cmd = [
        sys.executable, "-m", "uvicorn", "dashboard.backend.app:app",
        "--host", args.host, "--port", str(args.port),
    ]
    if args.reload:
        cmd += [
            "--reload",
            "--reload-dir", str(REPO_ROOT / "dashboard" / "backend"),
            "--reload-dir", str(REPO_ROOT / "multifunbrain"),
        ]
    ok(f"serving at {url}   (Ctrl-C to stop)")
    if args.host == "0.0.0.0":
        print(f"  on this network others can open:  http://<this-machine-ip>:{args.port}")
    child_env = {**os.environ, **env}
    if not args.no_open:
        # Open the browser shortly after the server comes up (best effort).
        try:
            import threading

            threading.Timer(2.0, lambda: webbrowser.open(url)).start()
        except Exception:  # noqa: BLE001
            pass
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=child_env)
    except KeyboardInterrupt:
        pass


# ── entry point ────────────────────────────────────────────────────────────
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="quickstart.py",
        description="One-command setup & launch for the multifun-brain dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python quickstart.py               full setup, then open the dashboard\n"
        "  python quickstart.py serve         just launch (everyday, after git pull)\n"
        "  python quickstart.py --dry-run     print the plan, change nothing\n"
        "  python quickstart.py --data /mnt/shared/multifun-data\n",
    )
    p.add_argument(
        "mode",
        nargs="?",
        choices=["setup", "serve"],
        default="setup",
        help="'setup' (default): install/ingest/build then serve. 'serve': launch only.",
    )
    p.add_argument("--data", type=Path, help="use a data folder that lives outside the repo")
    p.add_argument("--dry-run", action="store_true", help="show the plan; make no changes")
    p.add_argument("--no-ingest", action="store_true", help="don't compute missing bundles")
    p.add_argument("--rebuild", action="store_true", help="force a fresh frontend build")
    p.add_argument("--reinstall", action="store_true", help="force pip install even if present")
    p.add_argument("--no-serve", action="store_true", help="set up only; don't launch")
    p.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    p.add_argument("--reload", action="store_true", help="dev: auto-reload backend on edits")
    p.add_argument("--port", type=int, default=int(os.environ.get("MFB_DASHBOARD_PORT", 8000)))
    p.add_argument("--host", default=os.environ.get("MFB_DASHBOARD_HOST", "127.0.0.1"))
    return p.parse_args(argv)


def resolve_data_root(args: argparse.Namespace) -> tuple[Path, Path, dict[str, str]]:
    """Return (data_root, results_root, env-overrides) honouring --data / env vars."""
    if args.data:
        data_root = args.data.expanduser().resolve()
        if not data_root.is_dir():
            die(f"--data path does not exist: {data_root}")
        env = data_env(data_root)
        results_root = Path(env.get("MFB_RESULTS_ROOT", data_root / "correlation_matrices_results"))
        return data_root, results_root, env
    # Default: the repo's ./data, or an MFB_*-overridden layout already in the env.
    data_root = Path(os.environ.get("MFB_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
    results_root = Path(
        os.environ.get("MFB_RESULTS_ROOT", data_root / "correlation_matrices_results")
    ).expanduser().resolve()
    return data_root, results_root, {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    banner = "multifun-brain dashboard · quickstart"
    print(f"\n{_C['step'] if _COLOR else ''}{banner}{_C['end'] if _COLOR else ''}")
    if args.dry_run:
        warn("DRY RUN — nothing will be installed, computed, built or launched.")

    data_root, results_root, env = resolve_data_root(args)
    print(f"  data folder:   {data_root}" + ("" if data_root.is_dir() else "   (MISSING)"))
    print(f"  result bundles:{results_root}")

    ensure_python()

    if args.mode == "serve":
        # Everyday launch: assume set up already; just make sure a build exists.
        if not have("fastapi") or not have("multifunbrain"):
            die("dependencies missing — run full setup first:  python quickstart.py")
        if not ensure_frontend(args):
            return 1
        serve(env, args)
        return 0

    # Full setup.
    ensure_deps(args)
    if not data_root.is_dir():
        warn(
            f"data folder not found at {data_root}.\n"
            "    Obtain the shared 'schaefer' data folder from the maintainer and either\n"
            "    place it at ./data, or re-run with:  python quickstart.py --data /path/to/it\n"
            "    (See SETUP.md § 'Get the data'.) Continuing setup so the UI still builds."
        )
        inv: list[dict] = []
    else:
        inv = inventory(data_root, results_root)
        ensure_bundles(inv, results_root, env, args)

    built = ensure_frontend(args)

    if args.no_serve or not built:
        step("Setup complete")
        if not built:
            warn("no frontend build yet — install Node (see SETUP.md) then re-run.")
        else:
            ok("all set. Launch any time with:  python quickstart.py serve")
        return 0

    if args.dry_run:
        serve(env, args)  # prints the would-launch line
        step("Dry run complete — re-run without --dry-run to actually set up.")
        return 0

    serve(env, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
