"""Figure builders for the collaborator handoff.

Two builders, each taking ``(corr_raw, label, gamma, out_path)``:

- :func:`build_fig1`: 4 × 3 grid — descriptive network panels (raw / MP
  denoise / filter density / signed network / strength / clustering /
  percolation) plus a bottom row that summarises the LANS+LRG hierarchy
  (LANS network coloured by community at τ_min, dendrogram, partition
  stability index Ψ).
- :func:`build_fig2`: 5 × 3 grid — one row per k ∈ {2, 5, 7, 17, 31}.
  Columns are the LANS network coloured by the k-partition, Schaefer
  ortho-slice brain map, and dendrogram with a cut line at the height
  that yields k components.

Both builders are pure: they read no global state and write a single
PDF. ``07_make_three_figs.py`` exposes a CLI; ``09_handoff_batch.py``
calls them directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting
from scipy.cluster.hierarchy import dendrogram, fcluster, set_link_color_palette

from _fig_common import chain_pipeline, linkage_at_tau_min
from multifunbrain.analysis.lrg.partitions import partition_stability_index

PALETTE = [
    "#E41A1C", "#FF7F00", "#FFD700", "#ADFF2F", "#008000",
    "#00FA9A", "#00CED1", "#4682B4", "#0000CD", "#663399",
    "#FF00FF", "#FF1493",
    "#8B0000", "#D2691E", "#556B2F", "#2E8B57", "#008B8B",
    "#00008B", "#4B0082", "#8B008B",
]
n_palette = len(PALETTE)
SINGLETON_COLOR = "#B0BEC5"
GRAY_BRAIN_IDX = n_palette + 1

print("Fetching Schaefer atlas ...")
ATLAS = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
ATLAS_IMG = nib.load(ATLAS["maps"])
ATLAS_DATA = ATLAS_IMG.get_fdata().astype(int)
ATLAS_LABELS: list[str] = [
    b.decode("utf-8") if isinstance(b, bytes) else str(b)
    for b in ATLAS["labels"]
]
BRAIN_CUT_COORDS = (0, 0, 0)
DISCRETE_CMAP = ListedColormap([(1, 1, 1, 0)] + PALETTE + [SINGLETON_COLOR])

FIG2_K_VALUES: tuple[int, ...] = (2, 5, 7, 17, 31)

# Short codes for Yeo 17-network labels (used on fig 2 dendrogram leaves).
NETWORK_ABBR: dict[str, str] = {
    "Default": "Def",
    "Limbic": "Lim",
    "SalVentAttn": "SVA",
    "DorsAttn": "DorA",
    "Cont": "Cnt",
    "SomMot": "SM",
    "VisCent": "Vis",
    "VisPeri": "VisP",
    "TempPar": "TP",
}


def _abbreviate_roi_label(raw: str, atlas_idx: int) -> str:
    """Compact label like ``23·LH·DefB·PFCv1`` for a dendrogram leaf.

    ``atlas_idx`` is the Schaefer 0-based index of the parcel; the dot
    separator keeps the printed string short while staying readable.
    Unknown network names fall back to their first three letters.
    """
    s = raw
    if s.startswith("17Networks_"):
        s = s[len("17Networks_"):]
    parts = s.split("_")
    if len(parts) >= 2:
        hemi, net = parts[0], parts[1]
        suffix = ""
        net_root = net
        for sfx in ("A", "B", "C"):
            if net.endswith(sfx) and len(net) > 1:
                suffix = sfx
                net_root = net[:-1]
                break
        net_short = NETWORK_ABBR.get(net_root, net_root[:3]) + suffix
        rest = "".join(parts[2:])
        body = f"{hemi}·{net_short}·{rest}" if rest else f"{hemi}·{net_short}"
    else:
        body = s
    return f"{atlas_idx:02d}·{body}"


# ────────────────────────────────────────────────────────────────────
# Palette / colour-map helpers
# ────────────────────────────────────────────────────────────────────


def palette_color(cmap, label):
    idx = cmap.get(int(label))
    if idx is None:
        return SINGLETON_COLOR
    return PALETTE[idx % n_palette]


def assign_cmap_coarsest(part):
    """Coarsest level: each multi-node cluster gets a palette index assigned
    by size descending. Largest cluster → palette[0]."""
    sizes = {int(c): int((part == c).sum()) for c in np.unique(part)}
    multi = sorted([c for c, s in sizes.items() if s > 1], key=lambda c: -sizes[c])
    eligible = multi[:n_palette]
    cmap = {int(c): None for c in np.unique(part)}
    for i, c in enumerate(eligible):
        cmap[int(c)] = i
    return cmap


def split_cmap(coarse_p, fine_p, coarse_map, used):
    """Propagate colours coarse → fine. Largest fine sub-cluster inherits
    its parent's colour; siblings claim fresh palette slots."""
    new_map = {}
    used = set(used)
    this_stage = set()
    sizes_fine = {int(c): int((fine_p == c).sum()) for c in np.unique(fine_p)}

    fine_to_coarse = {}
    for c_fine in np.unique(fine_p):
        mask = fine_p == c_fine
        c_coarse = int(np.bincount(coarse_p[mask]).argmax())
        fine_to_coarse[int(c_fine)] = c_coarse

    largest_of = {}
    for c_coarse in np.unique(coarse_p):
        subs = [cf for cf, cc in fine_to_coarse.items() if cc == c_coarse]
        if subs:
            largest_of[int(c_coarse)] = max(subs, key=lambda c: sizes_fine[c])

    for c_fine in sorted(sizes_fine.keys(), key=lambda c: -sizes_fine[c]):
        if sizes_fine[c_fine] <= 1:
            new_map[int(c_fine)] = None
            continue
        if len(this_stage) >= n_palette:
            new_map[int(c_fine)] = None
            continue
        c_coarse = fine_to_coarse[c_fine]
        chosen = None
        if largest_of.get(c_coarse) == c_fine:
            parent_idx = coarse_map.get(c_coarse)
            if parent_idx is not None and parent_idx not in this_stage:
                chosen = parent_idx
        if chosen is None:
            chosen = next(
                (i for i in range(n_palette) if i not in this_stage), None
            )
        if chosen is None:
            new_map[int(c_fine)] = None
            continue
        new_map[int(c_fine)] = chosen
        this_stage.add(chosen)
        used.add(chosen)
    return new_map, used


def make_link_color_func(Z, partition, cluster_to_color, above_color):
    """``link_color_func`` for scipy ``dendrogram`` that colours each link
    by which fcluster cluster *all* its descendants belong to."""
    n = len(partition)
    cache = {}

    def cluster_of(node_id):
        if node_id in cache:
            return cache[node_id]
        if node_id < n:
            cid = int(partition[node_id])
            cache[node_id] = cid
            return cid
        row = node_id - n
        left = int(Z[row, 0])
        right = int(Z[row, 1])
        lc = cluster_of(left)
        rc = cluster_of(right)
        cid = lc if lc == rc else None
        cache[node_id] = cid
        return cid

    def link_color(node_id):
        cid = cluster_of(node_id)
        if cid is None:
            return above_color
        return cluster_to_color.get(int(cid), above_color)

    return link_color


def brain_image(partition_arr, colour_idx_for_label, node_list=None):
    """Build a NIfTI of the Schaefer parcellation with integer labels
    encoding the cluster palette index for each ROI."""
    out = np.zeros_like(ATLAS_DATA)
    if node_list is None:
        node_list = list(range(len(partition_arr)))
    pos_for_roi = {
        int(n): i for i, n in enumerate(node_list) if 0 <= int(n) < 100
    }
    for roi_idx in range(1, 101):
        atlas_zero_based = roi_idx - 1
        if atlas_zero_based not in pos_for_roi:
            ci = GRAY_BRAIN_IDX
        else:
            c = int(partition_arr[pos_for_roi[atlas_zero_based]])
            idx = colour_idx_for_label.get(c)
            if idx is None:
                ci = GRAY_BRAIN_IDX
            else:
                ci = (idx % n_palette) + 1
        out[ATLAS_DATA == roi_idx] = ci
    return nib.Nifti1Image(out.astype(np.int16), ATLAS_IMG.affine)


# ────────────────────────────────────────────────────────────────────
# Drawing primitives
# ────────────────────────────────────────────────────────────────────


def percolation_curve(W, n_points=80):
    W = np.abs(W).copy()
    np.fill_diagonal(W, 0.0)
    nz = W[np.triu_indices_from(W, k=1)]
    nz = nz[nz > 0]
    if len(nz) == 0:
        return np.array([0.0]), np.array([1.0]), np.array([1.0])
    thresholds = np.linspace(0.0, nz.max() * 0.999, n_points)
    n_total_edges = (W > 0).sum() // 2
    n_nodes = W.shape[0]
    p_inf, e_inf = [], []
    for th in thresholds:
        A = (W >= th).astype(int)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A)
        if G.number_of_edges() == 0:
            p_inf.append(0.0)
            e_inf.append(0.0)
            continue
        cc = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(cc)
        p_inf.append(len(cc) / n_nodes)
        e_inf.append(
            Gc.number_of_edges() / n_total_edges if n_total_edges else 0.0
        )
    return thresholds, np.array(p_inf), np.array(e_inf)


def _draw_positive_network(ax, G, pos, *, title, node_colors=None, gamma_lw=2.5):
    if G.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5, "Empty",
            ha="center", va="center", transform=ax.transAxes, color="gray",
        )
        ax.axis("off")
        ax.set_title(title)
        return
    edge_list = list(G.edges(data=True))
    wts = np.array([d.get("weight", 1.0) for _, _, d in edge_list])
    if wts.size:
        wmax = float(wts.max())
        x_str = (wts / wmax) ** gamma_lw
        widths = x_str * 2.8 + 0.10
        alphas = x_str * 0.75 + 0.05
        order = np.argsort(wts)
    else:
        widths = np.array([])
        alphas = np.array([])
        order = np.array([], dtype=int)
    for idx in order:
        u, v, _ = edge_list[idx]
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot(
                [x0, x1], [y0, y1],
                color="#37474F", lw=float(widths[idx]), alpha=float(alphas[idx]),
                solid_capstyle="round", zorder=1,
            )
    strength = dict(G.degree(weight="weight"))
    smax = max(strength.values()) if strength else 1.0
    node_list_loc = list(G.nodes())
    node_sizes = [60 + 220 * (strength[n] / smax) for n in node_list_loc]
    xs = np.array([pos[n][0] for n in node_list_loc])
    ys = np.array([pos[n][1] for n in node_list_loc])
    nc = node_colors if node_colors is not None else "#1565C0"
    ax.scatter(
        xs, ys,
        s=node_sizes, c=nc, edgecolors="white", linewidths=0.6, zorder=3,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_raw_signed_network(
    ax, corr_prep, pos, *, title, node_colors=None, gamma_lw=2.5
):
    """Draw the signed correlation network. Positive blue, negative red."""
    n = corr_prep.shape[0]
    np.fill_diagonal(corr_prep, 0.0)
    iu, ju = np.triu_indices(n, k=1)
    w = corr_prep[iu, ju]
    aw = np.abs(w)
    if aw.size == 0 or aw.max() == 0:
        ax.axis("off")
        ax.set_title(title)
        return
    wmax = aw.max()
    keep = aw >= 0.02 * wmax
    iu, ju, w, aw = iu[keep], ju[keep], w[keep], aw[keep]
    x_str = (aw / wmax) ** gamma_lw
    widths = x_str * 3.0 + 0.08
    alphas = x_str * 0.75 + 0.05

    def _draw_subset(mask, color):
        if not mask.any():
            return
        sub_idx = np.where(mask)[0]
        order = sub_idx[np.argsort(aw[sub_idx])]
        for k in order:
            i, j = int(iu[k]), int(ju[k])
            if i in pos and j in pos:
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                ax.plot(
                    [x0, x1], [y0, y1],
                    color=color, lw=float(widths[k]), alpha=float(alphas[k]),
                    solid_capstyle="round", zorder=1,
                )

    _draw_subset(w >= 0, "#1565C0")
    _draw_subset(w < 0, "#C62828")

    strength = np.abs(corr_prep).sum(axis=1)
    smax = strength.max() if strength.size else 1.0
    node_sizes = 60 + 220 * (strength / smax)
    pres = [i for i in range(n) if i in pos]
    xs = np.array([pos[i][0] for i in pres])
    ys = np.array([pos[i][1] for i in pres])
    ns = np.array([node_sizes[i] for i in pres])
    if node_colors is not None:
        nc = [node_colors[i] for i in pres]
    else:
        nc = "#37474F"
    ax.scatter(
        xs, ys, s=ns, c=nc, edgecolors="white", linewidths=0.6, zorder=3
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ────────────────────────────────────────────────────────────────────
# Fig 1 — combined network + LRG descriptive (4×3)
# ────────────────────────────────────────────────────────────────────


def build_fig1(corr_raw, label, gamma, out_path):
    """Twelve-panel collaborator figure. Top 3 rows = how the matrix was
    cleaned (raw / MP-denoise / filtering / strength / clustering /
    percolation), bottom row = LANS+LRG hierarchy summary (network
    coloured by community, dendrogram, partition-stability bars)."""
    from multifunbrain.processing.filtering import apply_all_filters

    corr_prep, corr_mp, corr_pos, G_lans = chain_pipeline(corr_raw, gamma)
    n = corr_prep.shape[0]
    N_lans = G_lans.number_of_nodes()

    fig, axes = plt.subplots(4, 3, figsize=(14.5, 18.0))
    fig.suptitle(
        f"Network & LRG descriptive  —  {label}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    mask = np.triu(np.ones_like(corr_prep, dtype=bool), k=1)
    bins = np.linspace(-1, 1, 60)

    # (0,0) raw signed corr
    im0 = axes[0, 0].imshow(
        corr_prep, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal"
    )
    axes[0, 0].set_title(f"Raw correlation  (n = {n})")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0,1) edge weight distribution
    ax_h = axes[0, 1]
    raw_vals = corr_prep[mask]
    mp_vals = corr_mp[mask]
    ax_h.hist(
        raw_vals[raw_vals < 0], bins=bins,
        color="#C62828", alpha=0.55, label="raw (−)",
    )
    ax_h.hist(
        raw_vals[raw_vals >= 0], bins=bins,
        color="#1565C0", alpha=0.55, label="raw (+)",
    )
    mp_counts, mp_edges = np.histogram(mp_vals, bins=bins)
    mp_centers = 0.5 * (mp_edges[1:] + mp_edges[:-1])
    ax_h.step(
        mp_centers, mp_counts, where="mid",
        color="#37474F", lw=1.5, label="MP-clean",
    )
    ax_h.axvline(0, color="k", lw=0.7, ls="--")
    ax_h.set_xlabel("Edge weight")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Edge weight distribution")
    ax_h.legend(fontsize=8, loc="upper left")

    # (0,2) MP-denoised heatmap
    im2 = axes[0, 2].imshow(
        corr_mp, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal"
    )
    axes[0, 2].set_title(f"After MP denoise (γ = {gamma:.3f})")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (1,0) filter density bar chart
    ax_fd = axes[1, 0]
    try:
        flt = apply_all_filters(
            corr_prep,
            methods=["absolute", "positive", "negative", "disparity", "lans"],
            threshold=None, alpha=0.05, gamma=gamma, sigma=1.0,
        )
    except Exception:
        flt = {}
    if flt:
        names = list(flt.keys())
        densities, nnodes = [], []
        for k in names:
            G = flt[k]["graph"]
            nn = G.number_of_nodes()
            ne = G.number_of_edges()
            d = ne / (nn * (nn - 1) / 2) if nn > 1 else 0.0
            densities.append(d)
            nnodes.append(nn)
        bar_cols = ["#546E7A", "#1565C0", "#C62828", "#2E7D32", "#6A1B9A"][
            : len(names)
        ]
        bars = ax_fd.bar(names, densities, color=bar_cols)
        for bar, nn in zip(bars, nnodes):
            ax_fd.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n = {nn}",
                ha="center", va="bottom", fontsize=8,
            )
        ax_fd.set_ylabel("Edge density")
        ax_fd.set_title("Filter density comparison")
        ax_fd.tick_params(axis="x", rotation=20, labelsize=8)
        if densities:
            ax_fd.set_ylim(0, min(1.05, max(densities) * 1.18))
    else:
        ax_fd.text(
            0.5, 0.5, "Filter computation failed",
            ha="center", va="center", transform=ax_fd.transAxes, color="gray",
        )
        ax_fd.axis("off")

    # (1,1) MP-clean signed network on KK / spring layout
    ax_net = axes[1, 1]
    G_signed = nx.from_numpy_array(np.abs(corr_mp))
    G_signed.remove_edges_from(list(nx.selfloop_edges(G_signed)))
    try:
        pos_signed = nx.kamada_kawai_layout(G_signed, weight="weight")
    except Exception:
        pos_signed = nx.spring_layout(
            G_signed, k=0.08, seed=42, iterations=200, weight="weight"
        )
    _draw_raw_signed_network(
        ax_net, corr_mp.copy(), pos_signed,
        title=f"MP-clean signed network  ({n} ROIs)",
    )

    # (1,2) MP-clean ∩ (weight > 0) heatmap
    vmax_pos = max(corr_pos.max(), 1e-6)
    im3 = axes[1, 2].imshow(
        corr_pos, cmap="Reds", vmin=0, vmax=vmax_pos, aspect="equal"
    )
    axes[1, 2].set_title("MP-clean ∩ (weight > 0)")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # (2,0) percolation
    ths, p_inf, e_inf = percolation_curve(corr_pos, n_points=120)
    max_per_node = corr_pos.max(axis=1)
    th_star = float(max_per_node.min()) if max_per_node.size else 0.0
    ax_p = axes[2, 0]
    ax_p.plot(ths, p_inf, color="#1565C0", lw=1.8, label=r"$P_\infty$")
    ax_p.plot(ths, e_inf, color="#C62828", lw=1.8, label=r"$E_\infty$")
    ax_p.axvline(
        th_star, color="k", ls="--", lw=0.8,
        label=fr"$\theta^*={th_star:.3f}$",
    )
    ax_p.set_xlabel("Threshold θ")
    ax_p.set_ylabel("Fraction")
    ax_p.set_title("Percolation curve")
    ax_p.set_xlim(0, max(ths.max(), 1e-6))
    ax_p.set_ylim(-0.02, 1.05)
    ax_p.legend(fontsize=8, loc="lower left")

    # (2,1) strength histogram
    strength_full = corr_pos.sum(axis=1)
    ax_s = axes[2, 1]
    if strength_full.size:
        ax_s.hist(
            strength_full, bins=22,
            color="#1565C0", edgecolor="white", alpha=0.85,
        )
        ax_s.axvline(
            strength_full.mean(), color="#C62828", ls="--", lw=1.1,
            label=f"mean = {strength_full.mean():.2f}",
        )
        ax_s.legend(fontsize=8)
    ax_s.set_xlabel("Node strength (Σ w)")
    ax_s.set_ylabel("Count")
    ax_s.set_title("Strength distribution")

    # (2,2) clustering histogram
    G_full = nx.from_numpy_array(corr_pos)
    G_full.remove_edges_from(list(nx.selfloop_edges(G_full)))
    if G_full.number_of_nodes() > 0:
        clustering = np.array(
            list(nx.clustering(G_full, weight="weight").values())
        )
    else:
        clustering = np.array([])
    ax_c = axes[2, 2]
    if clustering.size:
        ax_c.hist(
            clustering, bins=22,
            color="#2E7D32", edgecolor="white", alpha=0.85,
        )
        ax_c.axvline(
            clustering.mean(), color="#C62828", ls="--", lw=1.1,
            label=f"mean = {clustering.mean():.3f}",
        )
        ax_c.legend(fontsize=8)
    ax_c.set_xlabel("Weighted clustering coefficient")
    ax_c.set_ylabel("Count")
    ax_c.set_title("Clustering coefficient distribution")

    # Bottom row — LRG hierarchy summary on the LANS backbone.
    if N_lans >= 4:
        Z, leaf_atlas, tau = linkage_at_tau_min(G_lans)
        psi = partition_stability_index(Z)
        n_opt = int(np.argmax(psi)) if psi.size else 0
        sorted_d = np.sort(Z[:, 2])[::-1]
        if n_opt + 1 < len(sorted_d):
            upper = float(sorted_d[n_opt])
            lower = float(sorted_d[n_opt + 1])
            flat_th = (
                float(np.sqrt(upper * lower))
                if (upper > 0 and lower > 0)
                else 0.5 * (upper + lower)
            )
        else:
            flat_th = float(sorted_d[-1])
        partition = np.asarray(
            fcluster(Z, flat_th, criterion="distance"), dtype=int
        )

        leaf_order = [int(x) for x in dendrogram(Z, no_plot=True)["ivl"]]
        sizes_p = {
            int(c): int((partition == c).sum()) for c in np.unique(partition)
        }
        seen = []
        for leaf in leaf_order:
            c = int(partition[leaf])
            if c in seen or sizes_p[c] <= 1:
                continue
            seen.append(c)
        cluster_to_color = {
            c: PALETTE[i % len(PALETTE)] for i, c in enumerate(seen)
        }
        for c in np.unique(partition):
            if int(c) not in cluster_to_color:
                cluster_to_color[int(c)] = SINGLETON_COLOR

        pos_mds = nx.spring_layout(
            G_lans, k=0.08, seed=42, iterations=200, weight=None
        )
        node_colors_lans = [
            cluster_to_color[int(partition[i])] for i in range(N_lans)
        ]
        _draw_positive_network(
            axes[3, 0], G_lans, pos_mds,
            title=(
                fr"LANS backbone — {N_lans} nodes, "
                f"{G_lans.number_of_edges()} edges, "
                f"k = {len(set(partition))}"
            ),
            node_colors=node_colors_lans,
        )

        pal_for_dn = [cluster_to_color[c] for c in seen] or ["#1565C0"]
        set_link_color_palette(pal_for_dn)
        ax_d = axes[3, 1]
        dendrogram(
            Z, ax=ax_d, color_threshold=flat_th,
            above_threshold_color="#90A4AE", no_labels=True,
        )
        dpos = Z[:, 2]
        d_nz = dpos[dpos > 0]
        d_min = float(d_nz.min()) if d_nz.size else 1e-3
        d_max = float(dpos.max())
        ax_d.set_yscale("log")
        ax_d.set_ylim(d_min * 0.9, d_max * 1.1)
        ax_d.axhline(
            flat_th, color="#C62828", ls="--", lw=1,
            label=fr"flat threshold = {flat_th:.3f}",
        )
        ax_d.set_title(
            fr"LRG dendrogram   $\tau_\mathrm{{min}} = 1/\lambda_\mathrm{{max}} "
            fr"= {tau:.3g}$"
        )
        ax_d.set_xlabel("ROI")
        ax_d.set_ylabel("Diffusion distance (log)")
        ax_d.legend(fontsize=8)
        set_link_color_palette(None)

        ax_psi = axes[3, 2]
        psi_x = np.arange(len(psi))
        bar_colors = ["#1565C0"] * len(psi)
        if len(psi):
            bar_colors[n_opt] = "#C62828"
        ax_psi.bar(psi_x, psi, color=bar_colors)
        ax_psi.set_xlabel("Dendrogram branch (k = 2 … N)")
        ax_psi.set_ylabel("Ψ")
        ax_psi.set_title("Partition stability index (Ψ)")
        if len(psi):
            ax_psi.axvline(
                n_opt, color="#C62828", ls="--", lw=0.7,
                label=f"optimal k = {n_opt + 2}",
            )
            ax_psi.legend(fontsize=8)
    else:
        for j in range(3):
            axes[3, j].text(
                0.5, 0.5, "LANS backbone too small for LRG",
                ha="center", va="center",
                transform=axes[3, j].transAxes, color="gray",
            )
            axes[3, j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Fig 2 — dendrogram cuts at fixed k
# ────────────────────────────────────────────────────────────────────


def _dendro_cut_scan_fixed_k(corr_raw, gamma, ks):
    """Build the τ_min dendrogram once and cut at each k in ``ks`` via
    ``fcluster(Z, k, criterion='maxclust')``. Returns ``None`` if the LANS
    backbone is too small.

    The cut **height** annotated on the dendrogram is the geometric mean
    of the two merge distances that bracket the k-cut (mid-gap on the
    log-axis), so a horizontal line at that height yields exactly k
    components.

    Colour propagation: assign palette to the **coarsest** row by size
    descending, then split outward — sibling clusters with the largest
    overlap inherit the parent's slot."""
    _, _, _, G_lans = chain_pipeline(corr_raw, gamma)
    N = G_lans.number_of_nodes()
    if N < 5:
        return None
    Z, leaf_atlas, tau = linkage_at_tau_min(G_lans)

    sorted_d = np.sort(Z[:, 2])
    if sorted_d.size < 2:
        return None

    def mid_gap_for_k(k: int) -> float:
        if k >= N:
            return float(sorted_d[0]) / float(np.sqrt(2.0))
        idx_lower = N - k - 1
        idx_upper = N - k
        if 0 <= idx_lower < idx_upper < sorted_d.size:
            lo = float(sorted_d[idx_lower])
            hi = float(sorted_d[idx_upper])
            if lo > 0 and hi > 0:
                return float(np.sqrt(lo * hi))
            return 0.5 * (lo + hi)
        return float(sorted_d[-1]) * float(np.sqrt(2.0))

    ks_use = [int(k) for k in ks if 2 <= int(k) < N]
    if not ks_use:
        return None
    results = []
    for k in ks_use:
        part = np.asarray(fcluster(Z, int(k), criterion="maxclust"), dtype=int)
        results.append(
            {
                "k_target": int(k),
                "partition": part,
                "k_actual": int(len(np.unique(part))),
                "height": mid_gap_for_k(int(k)),
            }
        )

    # Coarsest first → fine: propagate so larger groups keep their colour.
    results_sorted_idx = sorted(
        range(len(results)), key=lambda i: results[i]["k_actual"]
    )
    cmap_per_row: list[dict | None] = [None] * len(results)
    cmap_last = assign_cmap_coarsest(results[results_sorted_idx[0]]["partition"])
    cmap_per_row[results_sorted_idx[0]] = cmap_last
    used = {v for v in cmap_last.values() if v is not None}
    for i in range(1, len(results_sorted_idx)):
        prev_pos = results_sorted_idx[i - 1]
        this_pos = results_sorted_idx[i]
        new_map, used = split_cmap(
            results[prev_pos]["partition"],
            results[this_pos]["partition"],
            cmap_per_row[prev_pos],
            used,
        )
        cmap_per_row[this_pos] = new_map

    return {
        "G_lans": G_lans,
        "N": N,
        "node_list": list(G_lans.nodes()),
        "Z": Z,
        "tau": tau,
        "results": results,
        "colour_maps": cmap_per_row,
        "leaf_atlas": leaf_atlas,
    }


def build_fig2(corr_raw, label, gamma, out_path, ks: tuple[int, ...] = FIG2_K_VALUES):
    """Dendrogram cuts at fixed k. One row per ``k`` in ``ks``; three
    columns: LANS network coloured by the k-partition, brain ortho-slices,
    dendrogram with the cut line."""
    data = _dendro_cut_scan_fixed_k(corr_raw, gamma, ks)
    if data is None:
        return
    G_lans = data["G_lans"]
    N = data["N"]
    node_list = data["node_list"]
    Z = data["Z"]
    tau = data["tau"]
    results = data["results"]
    colour_maps = data["colour_maps"]

    pos_spring = nx.spring_layout(
        G_lans, k=0.08, seed=42, iterations=200, weight=None
    )

    wts = np.array(
        [d.get("weight", 1.0) for _, _, d in G_lans.edges(data=True)]
    )
    wmax = wts.max() if len(wts) else 1.0
    edge_widths = wts / wmax * 1.2 + 0.06
    edge_alphas = wts / wmax * 0.5 + 0.05
    strength = dict(G_lans.degree(weight="weight"))
    smax = max(strength.values()) if strength else 1.0
    node_sizes = [60 + 220 * (strength[node_list[i]] / smax) for i in range(N)]

    dpos = Z[:, 2]
    d_nz = dpos[dpos > 0]
    d_y_lo = float(d_nz.min()) if d_nz.size else 1e-3
    d_y_hi = float(dpos.max())

    leaf_labels = [
        _abbreviate_roi_label(ATLAS_LABELS[int(node_list[i])], int(node_list[i]))
        if 0 <= int(node_list[i]) < len(ATLAS_LABELS)
        else f"{int(node_list[i]):02d}"
        for i in range(N)
    ]

    n_rows = len(results)
    fig = plt.figure(figsize=(22, 3.8 * n_rows + 0.8))
    gs = gridspec.GridSpec(
        n_rows, 3, figure=fig,
        height_ratios=[1] * n_rows,
        width_ratios=[0.7, 1.3, 2.0],
        hspace=0.55, wspace=0.10,
        top=0.965, bottom=0.030, left=0.025, right=0.99,
    )
    fig.suptitle(
        fr"Dendrogram cuts at fixed k    "
        fr"($\tau_\mathrm{{min}} = {tau:.3g}$)  —  {label}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    for r in range(n_rows):
        info = results[r]
        part = info["partition"]
        cmap_r = colour_maps[r]
        h = info["height"]
        k_target = info["k_target"]
        k_actual = info["k_actual"]

        # col 0 — LANS network coloured by partition.
        ax_net = fig.add_subplot(gs[r, 0])
        for (u, v, _), w, a in zip(
            G_lans.edges(data=True), edge_widths, edge_alphas
        ):
            nx.draw_networkx_edges(
                G_lans, pos_spring, edgelist=[(u, v)],
                ax=ax_net, width=float(w),
                edge_color="#37474F", alpha=float(a),
            )
        node_cols = [palette_color(cmap_r, part[i]) for i in range(N)]
        nx.draw_networkx_nodes(
            G_lans, pos_spring, ax=ax_net,
            node_size=node_sizes, node_color=node_cols,
            edgecolors="white", linewidths=0.6,
        )
        title = fr"k = {k_actual}"
        if k_actual != k_target:
            title += f"  (asked k = {k_target})"
        ax_net.set_title(title, fontsize=11)
        ax_net.set_xticks([])
        ax_net.set_yticks([])
        for s in ax_net.spines.values():
            s.set_visible(False)

        # col 1 — brain ortho.
        ax_brain = fig.add_subplot(gs[r, 1])
        img = brain_image(part, cmap_r, node_list=node_list)
        plotting.plot_roi(
            img, axes=ax_brain, display_mode="ortho",
            cut_coords=BRAIN_CUT_COORDS,
            cmap=DISCRETE_CMAP, colorbar=False,
            vmin=0, vmax=n_palette + 1,
            annotate=False, draw_cross=False, alpha=0.9, black_bg=False,
        )
        ax_brain.set_title("Schaefer-100 parcels", fontsize=10)

        # col 2 — dendrogram with cut line and abbreviated leaf labels.
        ax_dn = fig.add_subplot(gs[r, 2])
        cluster_to_color = {
            int(c): palette_color(cmap_r, c) for c in np.unique(part)
        }
        link_color = make_link_color_func(
            Z, part, cluster_to_color, SINGLETON_COLOR
        )
        dendrogram(
            Z, ax=ax_dn, link_color_func=link_color,
            labels=leaf_labels,
            leaf_rotation=90, leaf_font_size=5,
        )
        ax_dn.set_yscale("log")
        ax_dn.set_ylim(d_y_lo * 0.85, d_y_hi * 1.15)
        ax_dn.axhline(h, color="#C62828", ls="--", lw=1)
        ax_dn.set_title(fr"Dendrogram   cut $h$ = {h:.3g}", fontsize=11)
        ax_dn.set_ylabel("Diff. dist. (log)", fontsize=9)
        ax_dn.tick_params(axis="y", labelsize=8)

    set_link_color_palette(None)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "FIG2_K_VALUES",
    "build_fig1",
    "build_fig2",
]
