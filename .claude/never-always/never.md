# Never

## Never replace NaN with 0 in a correlation matrix

**Why:** NaN rows/columns indicate dead brain regions (zero-variance
fMRI time series). Replacing with 0 fabricates correlation values that
do not exist and silently contaminates every downstream analysis.

**How to apply:** Use `multifunbrain.preprocessing.detect_dead_regions`
to identify them, then **drop** the rows/columns
(`prepare_correlation_matrix` does this automatically). Record the
dropped indices in `PipelineResult.dropped_regions`.

---

## Never commit data files

**Why:** Pickles and atlas files are large, often private, and
non-reproducible from the repo. `.gitignore` has `data/*` precisely
for this.

**How to apply:** Always inspect `git status` before `git add`. Prefer
adding files by name; avoid `git add -A` or `git add .` in directories
that may contain collaborator data. If you accidentally stage data, run
`git restore --staged <path>` before committing.

---

## Never run destructive git operations without explicit user approval

**Why:** `git reset --hard`, `git push --force`, `git branch -D`,
`git checkout .`, `git clean -f` can permanently destroy work — including
work the user has not yet inspected.

**How to apply:** If you genuinely need one of these, ask first. State
the alternative ("instead of reset --hard we can stash"), what's at
risk, and what the recovery would look like.

---

## Never amend or rewrite published commits

**Why:** Other contributors may have based work on those commits; force-
pushing rewrites breaks their history and triggers conflict cascades.

**How to apply:** Always create a new commit rather than `--amend`,
especially after a hook fails (the hook failure means the previous
commit didn't happen the way you expected; `--amend` would mutate the
*prior* commit instead).

---

## Never skip hooks (`--no-verify`, `--no-gpg-sign`)

**Why:** Hooks exist to protect the repo (formatting, lint, tests, sign-
offs). Skipping them ships broken code.

**How to apply:** If a hook fails, fix the issue. If a hook is wrong,
fix the hook. Bypass only when the user explicitly says so.

---

## Never unpickle collaborator files outside the analysis pipeline

**Why:** Pickle deserialisation executes arbitrary code. The sandbox
correctly blocks ad-hoc `pickle.load` on data directories. The
intentional, documented call site (the pipeline's `load_correlation_matrix`)
is the only place we accept that risk.

**How to apply:** To inspect file structure, peek at how the existing
loaders handle the format (read code, not pickle bytes). Pickle reads
happen inside library functions, not in throwaway sandbox commands.

---

## Never inline analysis logic in a notebook

**Why:** Logic in notebooks is not testable, not reusable, and silently
drifts as cells are run out of order.

**How to apply:** New analytical function → goes in
`multifunbrain/<scope>/`. New plot → goes in
`multifunbrain/visualization/plotlib/`. Notebook only loads
pre-computed results and calls library functions.

---

## Never drop the back-compat shim files in `multifunbrain/analysis/{corrmatrix,descriptive,filtering,netmetrics,lrglib}.py` without a separate migration PR

**Why:** Existing external notebooks and scripts import from these
paths. Removing them silently breaks downstream work.

**How to apply:** Removing a shim requires its own PR that first audits
all callers (`rg "from multifunbrain.analysis.<file>" `) and migrates
them to canonical paths. Land that before deleting the shim.

---

## Never duplicate a function definition across files

**Why:** Two copies drift apart, fixes land in only one, downstream
behaviour becomes locally inconsistent. The user explicitly flagged
this rule.

**How to apply:** A function has *one* canonical home. Other modules
must `import` it, not redefine it. If you find yourself copy-pasting,
stop — promote the helper to a shared module instead.

---

## Never clutter source code with long agentic comments

**Why:** Code is read more often than written; long decision-context
or *"see .claude/..."* commentary turns source files into a wiki. The
agentic folder (`.claude/{guide,history,never-always,reports}/` plus
auto-memory) is the single home for rationale and project history.

**How to apply:** Inside `multifunbrain/`, `scripts/`, `test/`, and
notebooks, comments and docstrings must serve **human comprehension
only** — what a parameter means, units, non-obvious math, a paper
reference. No multi-line *"defaults are deliberately ..."* blocks, no
references to `.claude/` paths from source files. When the rationale
matters, write it in `.claude/history/YYYY-MM-DD_<topic>.md` or in
the relevant `.claude/guide/` page.

---

## Never tell the user to run a notebook or script you haven't executed

**Why:** The user reported wasted time after opening a notebook that
errored on the first plotting cell (`plot_lrg_entropy() got an
unexpected keyword argument 'fname'`). API signatures recalled from
memory or pattern-matched from prior code are unreliable.

**How to apply:** Before saying *"open notebook X"* or *"run script
X.py"*, the assistant **must** execute it end-to-end and confirm zero
error outputs. For notebooks:
``cd notebooks/<dir> && jupyter nbconvert --to notebook --execute --inplace <file>.ipynb``
— **always run nbconvert with the same CWD the user will have**
(typically the notebook's own directory), not the repo root.
Relative paths inside a notebook resolve against the kernel's CWD,
and verifying from the wrong CWD silently masks broken paths
(`0/40 matrices loaded` instead of an error). For scripts: run them.
For code snippets pasted in a reply: run them. If the environment
can't support a real execution (no data, no kernel, etc.), say so
explicitly — never hand-wave that the artefact "should work". See
`.claude/guide/plotting.md` for the canonical verification snippet.

---

## Never inline plot composition or analysis logic in notebooks

**Why:** Loops like `for r in results: fig, ax = plt.subplots(...)`
create visual clutter, are not reusable, drift when the data shape
changes, and bury logic where tests can't reach it.

**How to apply:** Plot patterns belong in
`multifunbrain.visualization.plotlib.*`. The canonical multi-panel
helper is `plot_results_grid` — use it instead of writing loops in a
notebook. New plot patterns get a helper added to `pipeline_plots.py`
first, then a one-line call from the notebook. See
`.claude/guide/plotting.md` for the full rule set.
