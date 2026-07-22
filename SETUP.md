# Getting the dashboard running — complete guide (Windows · macOS · Linux)

This is the full walk-through, **from cloning the repo to seeing the dashboard in
your browser**. It is identical on Windows, macOS and Linux — every command below
is cross-platform.

The whole thing is **one command after setup**:

```bash
python quickstart.py
```

That single script installs the dependencies, **auto-detects your data folder**
(atlas, subjects, processing variants), **computes and caches** any results that
aren't there yet, builds the interface, and opens the dashboard in your browser.

---

## The 5 steps at a glance

```bash
# 1. Clone
git clone https://github.com/giulioiannelli/multifun-brain.git
cd multifun-brain

# 2. Create the environment (Python + Node in one step, all three OSes)
conda env create -f multifun-brain.yml
conda activate multifun-brain

# 3. Get the data  →  place the shared folder at ./data   (see §3)

# 4. Set up + launch (first time: installs, detects data, caches results, builds UI)
python quickstart.py

# 5. Your browser opens http://localhost:8000  — that's it.
```

Every day after that, and every time you `git pull` new features:

```bash
conda activate multifun-brain
python quickstart.py serve        # just launches — instant
```

---

## 1. Prerequisites

You need **one** thing: **Conda** (Miniconda or Anaconda). It installs Python
**and** Node.js for you, the same way on every OS, so there is nothing else to
chase down.

- Install Miniconda: <https://docs.conda.io/en/latest/miniconda.html>
- Windows: use the **"Anaconda Prompt"** (or "Miniconda Prompt") from the Start
  menu for every command below. macOS/Linux: any terminal.

<details>
<summary>Prefer a plain <code>venv</code> instead of conda?</summary>

That works too, but then **you install Node.js ≥ 18 yourself**
(<https://nodejs.org>), because the frontend is built with it. After that:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS/Linux:    source .venv/bin/activate
python quickstart.py           # quickstart runs the pip install for you
```
</details>

## 2. Clone and enter the project

```bash
git clone https://github.com/giulioiannelli/multifun-brain.git
cd multifun-brain
```

The repository contains **code only** — the brain data is large and private, so
it lives outside git (next step).

## 3. Get the data

The `data/` folder is **not** on GitHub (it's big and private). Obtain the shared
data folder from the maintainer (via the shared drive / link they gave you) and
put it at the **repository root** so the layout is:

```
multifun-brain/
├── quickstart.py
├── data/                                   ← the shared folder goes here
│   ├── correlation_matrices_results/       ← computed results (if shipped)
│   ├── raw_data_schaefer100_april2026/     ← raw timecourses, one folder per subject
│   ├── raw_data_schaefer100_november2025/
│   ├── raw_data_harvardoxford48/
│   ├── schaefer_2018/                      ← atlas (region names + 3-D parcellation)
│   └── HarvardOxford-Cortical.xml          ← atlas labels for the HarvardOxford set
└── ...
```

Two cases, both handled automatically by `quickstart.py`:

- **The folder already includes `correlation_matrices_results/`** (the computed
  bundles). → quickstart detects them and **skips straight to building + serving**.
  This is the fastest way to *look at results*.
- **The folder has only the `raw_data_*` timecourses.** → quickstart **computes
  and caches** the result bundles for you the first time (the one slow step; see
  §5). Later runs reuse the cache.

> **Keeping the data somewhere else?** If you'd rather not move a big folder into
> the repo, leave it where it is and point quickstart at it:
> ```bash
> python quickstart.py --data /path/to/the/shared/folder
> ```
> Nothing is copied — the server is simply pointed at that location.

You don't need to rename or reorganise anything inside the folder. quickstart
**recognises the structure itself** — which atlas each `raw_data_<id>` set uses,
how many subjects it has, and which processing variants exist.

## 4. Set up and launch

```bash
python quickstart.py
```

On the **first** run this will, in order, printing what it's doing:

1. **Check Python** and **install** the project + dashboard dependencies (skipped
   automatically if the conda env already has them).
2. **Inventory your data** — prints a table like:

   ```
   dataset                      atlas            subj vars  bundles
   ---------------------------- ---------------- ---- ----  ------------------------
   harvardoxford48              HarvardOxford-48   13    4  global·bands·patients
   schaefer100_april2026        Schaefer-100        6    5  global·bands·patients
   schaefer100_november2025     Schaefer-100       11   10  global·bands·patients
   ```
3. **Compute + cache** the bundles for any dataset whose `bundles` column shows
   "— none" (skipped when everything is already there — as above).
4. **Build the interface** (only the first time, or with `--rebuild`).
5. **Serve** on `http://localhost:8000` and **open your browser**.

**Want to see the plan before anything happens?** Add `--dry-run`:

```bash
python quickstart.py --dry-run     # prints exactly what it *would* do; changes nothing
```

## 5. First-run compute time (only if results aren't shipped)

If your data folder had **only raw timecourses**, quickstart computes the result
bundles once. This is genuinely heavy (Empirical-Mode-Decomposition over every
brain region plus the multiscale LRG analysis for every matrix): **minutes to a
few hours per dataset**, depending on the machine. It runs unattended and prints
progress with a running ETA. When it finishes, the results are cached under
`data/correlation_matrices_results/<dataset>/` and **never recomputed** — so this
cost is paid exactly once.

- To **skip** computing (e.g. you only care about datasets that are already
  bundled): `python quickstart.py --no-ingest`.
- If the maintainer can share the pre-computed `correlation_matrices_results/`
  folder, drop it in and this step is skipped entirely.

## 6. Open the dashboard

quickstart opens your default browser at **http://localhost:8000** automatically.
If it doesn't (or you passed `--no-open`), just open that address yourself. Press
**Ctrl-C** in the terminal to stop the server.

You'll see tabs for **Signal** (raw timecourses · carpet · EMD · bands),
**Correlation**, **Network**, **Brain-3D**, **LRG** (multiscale), and a
**Pipeline** methodology view.

---

## Staying up to date (pull new features)

The maintainer keeps adding to the dashboard. To get the latest:

```bash
git pull
conda activate multifun-brain
python quickstart.py serve
```

- `serve` launches immediately — no rebuild, no recompute.
- If a pull changed the **frontend**, force a one-off rebuild: `python
  quickstart.py --rebuild` (or just `python quickstart.py`, which rebuilds when
  needed).
- If the maintainer added a **new dataset**, run the full `python quickstart.py`
  once — it detects the new `raw_data_*` folder and caches its bundles, leaving
  the existing ones untouched.

## Let non-coders "just open a URL" (one host, zero install for everyone else)

Run the dashboard on **one** machine and share it with collaborators on the same
network/VPN — they need **only a browser**:

```bash
python quickstart.py serve --host 0.0.0.0
```

The host machine prints the address to share (e.g. `http://192.168.1.42:8000`).
Find the host's IP with `ipconfig` (Windows) or `hostname -I` (macOS/Linux) if
needed, and allow inbound TCP on the port through the host firewall.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **"port 8000 is already in use"** | A previous server is still running. Use another port: `python quickstart.py serve --port 8001`. |
| **The page never loads / spins forever** | Almost always the port issue above — pick another port. |
| **"Node.js/npm not found and no existing build"** | The interface can't be built. Use the conda env (`conda env create -f multifun-brain.yml`), which bundles Node — or install Node ≥ 18 and re-run. |
| **"data folder not found"** | Put the shared folder at `./data` (§3), or pass `--data /path/to/it`. |
| **Result tabs are empty, Signal tab works** | The result bundles weren't computed. Run `python quickstart.py` (without `--no-ingest`) to compute them, or get the pre-computed `correlation_matrices_results/` from the maintainer. |
| **Windows: `./quickstart.py` "not recognized"** | Always call it as `python quickstart.py` (the `python` prefix matters on Windows). |
| **Windows: `run.sh` doesn't work** | `run.sh` is for macOS/Linux (and WSL/Git-Bash). On native Windows use `python quickstart.py` — it does the same thing cross-platform. |
| **A dataset shows some bands as "failed"** | Expected: high-frequency bands above a slow acquisition's Nyquist limit are recorded as graceful failures (`failed_matrices.json`), not errors. |

## Command reference

```
python quickstart.py                 full setup, then open the dashboard
python quickstart.py serve           just launch (everyday use, after git pull)
python quickstart.py --dry-run       show the plan; make no changes
python quickstart.py --data PATH     use a data folder that lives outside the repo
python quickstart.py --no-ingest     don't compute missing result bundles
python quickstart.py --rebuild       force a fresh interface build
python quickstart.py --no-serve      set up only; don't launch
python quickstart.py --port 8001     serve on a different port
python quickstart.py --host 0.0.0.0  share on the local network
python quickstart.py --reload        dev: auto-reload the backend on code edits
```

For the deeper reference — every data location, the environment-variable
overrides, the exact result-bundle label scheme, and the development (hot-reload)
workflow — see **[`dashboard/README.md`](dashboard/README.md)**.
