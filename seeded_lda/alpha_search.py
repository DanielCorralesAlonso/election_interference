import os
import json
import numpy as np
import tomotopy as tp
import matplotlib.pyplot as plt
from tqdm import tqdm

from set_seeded_prior import set_seeded_prior
from find_best_k import _topic_diversity


def _mean_doc_topic_entropy(model):
    """Mean normalised Shannon entropy of theta_d across documents.

    H(theta_d) / log(K): 1.0 = uniform over topics (diffuse), 0 = single
    dominant topic per document (peaky). Directly measures what alpha controls.
    """
    log_k = np.log(model.k)
    entropies = []
    for doc in model.docs:
        theta = np.array(doc.get_topic_dist())
        theta = theta[theta > 0]
        entropies.append(-np.sum(theta * np.log(theta)) / log_k)
    return float(np.mean(entropies))


def _mean_doc_top_topic_share(model):
    """Mean of max_k(theta_d) across documents — how dominated each document
    is by its single most probable topic. Complements entropy."""
    return float(np.mean([max(doc.get_topic_dist()) for doc in model.docs]))


def run_alpha_search(documents, alpha_values, k, eta=0.01, min_cf=0, tw=None,
                     n_iterations=4000, coherence_measure="c_v", top_n=20,
                     use_diversity=True, diversity_top_n=20,
                     seed_lexicon=None, seed_weight=1.0, regular_weight=0.0,
                     output_dir="output", country_name="all", random_seed=None,
                     n_train_workers=0, optim_interval=None):
    """
    Train seeded-LDA models at fixed K over a range of alpha values and report,
    for each: C_V coherence (overall, and split seeded vs. unseeded topics),
    topic diversity, mean normalised document-topic entropy, mean top-topic
    share, and perplexity.

    alpha (the document-topic Dirichlet concentration) controls how "peaky"
    each document's topic distribution is — low alpha pushes documents toward
    a single dominant topic, high alpha toward diffuse mixtures. The entropy
    and top-topic-share panels make that effect directly visible, alongside
    the usual quality metrics (coherence, diversity), so you can see whether
    a given alpha is buying topic quality or just forcing sharper assignments.

    optim_interval : int or None
        Forwarded to `tp.LDAModel`. Tomotopy periodically re-estimates alpha
        (and eta) from the data via a fixed-point method every
        `optim_interval` iterations — the alpha you pass in is then only an
        *initial* value, not the value the model actually trains with.
        Pass `optim_interval=0` to disable this and test a genuinely fixed
        alpha; pass `None` to use tomotopy's own default (auto-optimisation
        on). Whenever optimisation is active (`optim_interval != 0`), the
        post-training alpha actually used by each model is logged and
        plotted alongside the initial value, so you can see whether your
        sweep is measuring the prior you set or the one tomotopy converged to.

    Returns
    -------
    best_alpha : float — alpha with highest overall C_V coherence
    best_model : trained LDAModel at best_alpha
    results : dict — maps each alpha to its metrics
    """
    if tw is None:
        tw = tp.TermWeight.IDF

    alpha_list = list(alpha_values)
    optimization_active = optim_interval != 0

    topic_name_to_id = (
        {name: i for i, name in enumerate(seed_lexicon.keys())}
        if seed_lexicon else {}
    )
    seeded_topic_ids = set(topic_name_to_id.values())
    unseeded_topic_ids = [i for i in range(k) if i not in seeded_topic_ids]

    print(f"\nSweeping alpha over {alpha_list} (K={k}, {n_iterations} iterations each, "
          f"hyperparameter optimisation {'ON (tomotopy default)' if optim_interval is None else ('ON every ' + str(optim_interval) + ' iters' if optim_interval else 'OFF — fixed alpha')})...")

    def _train_alpha(alpha_val):
        m = tp.LDAModel(k=k, alpha=alpha_val, eta=eta, min_cf=min_cf, tw=tw, seed=random_seed)
        if optim_interval is not None:
            # Not a constructor argument in tomotopy — set as an instance
            # attribute before training to control hyperparameter
            # re-estimation frequency (0 disables it).
            m.optim_interval = optim_interval
        for doc in documents:
            m.add_doc(doc)
        if seed_lexicon:
            set_seeded_prior(m, seed_lexicon, topic_name_to_id,
                             seed_weight=seed_weight, regular_weight=regular_weight,
                             verbose=False)
        m.train(n_iterations, workers=n_train_workers)

        # tomotopy may have re-estimated alpha during training (per-topic
        # vector) — log what the model actually ended up using vs. the
        # initial value you requested.
        final_alpha_arr = np.atleast_1d(np.array(m.alpha, dtype=float))
        final_alpha_mean = float(np.mean(final_alpha_arr))
        final_alpha_std = float(np.std(final_alpha_arr))

        coh_eval = tp.coherence.Coherence(m, coherence=coherence_measure, top_n=top_n)
        topic_scores = [coh_eval.get_score(topic_id=i) for i in range(m.k)]
        avg_coh = float(np.mean(topic_scores))
        seeded_coh = (float(np.mean([topic_scores[i] for i in seeded_topic_ids]))
                      if seeded_topic_ids else None)
        unseeded_coh = (float(np.mean([topic_scores[i] for i in unseeded_topic_ids]))
                        if unseeded_topic_ids else None)

        div = _topic_diversity(m, top_n=diversity_top_n) if use_diversity else None
        entropy = _mean_doc_topic_entropy(m)
        top_share = _mean_doc_top_topic_share(m)

        return {
            "alpha": alpha_val,
            "model": m,
            "coherence": avg_coh,
            "coherence_seeded": seeded_coh,
            "coherence_unseeded": unseeded_coh,
            "diversity": div,
            "doc_topic_entropy": entropy,
            "doc_top_topic_share": top_share,
            "perplexity": float(m.perplexity),
            "final_alpha_mean": final_alpha_mean,
            "final_alpha_std": final_alpha_std,
        }

    def _log_line(rec):
        line = (f"  alpha={rec['alpha']:.2e}  coherence={rec['coherence']:.4f}"
                f"  (seeded={rec['coherence_seeded']:.4f}, unseeded={rec['coherence_unseeded']:.4f})"
                f"  entropy={rec['doc_topic_entropy']:.4f}  top_share={rec['doc_top_topic_share']:.4f}")
        if optimization_active:
            line += (f"  -> post-train alpha (mean+-std) = "
                     f"{rec['final_alpha_mean']:.4g} +- {rec['final_alpha_std']:.4g}")
        return line

    records = []
    for alpha_val in tqdm(alpha_list, desc="Alpha search"):
        rec = _train_alpha(alpha_val)
        records.append(rec)
        tqdm.write(_log_line(rec))

    if optimization_active:
        initial = np.array(alpha_list, dtype=float)
        final = np.array([r["final_alpha_mean"] for r in records], dtype=float)
        spread = final.max() - final.min()
        print(f"\nHyperparameter optimisation was ACTIVE "
              f"(optim_interval={'tomotopy default' if optim_interval is None else optim_interval}).")
        print(f"  Initial alphas spanned {initial.min():.2e} .. {initial.max():.2e} "
              f"({initial.max() / max(initial.min(), 1e-300):.1e}x range)")
        print(f"  Post-training mean alphas spanned {final.min():.4g} .. {final.max():.4g} "
              f"(absolute spread {spread:.4g})")
        if spread < 0.5 * (final.mean()):
            print("  -> Post-training alphas converged to a similar value regardless of the "
                  "initial value: the sweep below largely reflects tomotopy's data-driven "
                  "alpha estimate, NOT the alpha you requested. Re-run with optim_interval=0 "
                  "to test a genuinely fixed alpha.")

    coh_arr = np.array([r["coherence"] for r in records])
    best_idx = int(np.argmax(coh_arr))
    best_alpha = records[best_idx]["alpha"]
    best_model = records[best_idx]["model"]

    print(f"\nAlpha selection summary (criterion: C_V coherence, K={k}):")
    for r in records:
        marker = " <--" if r["alpha"] == best_alpha else ""
        diversity_str = f"  diversity={r['diversity']:.4f}" if r['diversity'] is not None else ""
        final_alpha_str = (f"  post-train alpha={r['final_alpha_mean']:.4g}+-{r['final_alpha_std']:.4g}"
                           if optimization_active else "")
        print(f"  alpha={r['alpha']:.2e}  coherence={r['coherence']:.4f}"
              f"  seeded={r['coherence_seeded']:.4f}  unseeded={r['coherence_unseeded']:.4f}"
              f"{diversity_str}"
              f"  entropy={r['doc_topic_entropy']:.4f}  top_share={r['doc_top_topic_share']:.4f}"
              f"  perplexity={r['perplexity']:.4f}{final_alpha_str}{marker}")
    print(f"Best alpha: {best_alpha:.2e}  (C_V = {coh_arr[best_idx]:.4f})")

    # --- plot ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    n_cols = 3 if optimization_active else 2
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))

    def _panel(ax, y, ylabel, title, color="darkorange", extra=None):
        ax.semilogx(alpha_list, y, marker="o", color=color, label=ylabel)
        if extra is not None:
            for label, vals, c in extra:
                ax.semilogx(alpha_list, vals, marker="s", linestyle="--", color=c, label=label)
        ax.axvline(best_alpha, color="red", linestyle="--", label=f"selected alpha={best_alpha:.1e}")
        ax.set_xlabel("alpha (log scale)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8)

    _panel(axes[0, 0],
           [r["coherence"] for r in records],
           f"{coherence_measure} coherence",
           "Coherence vs. alpha\n(overall, seeded, unseeded)",
           color="darkorange",
           extra=[
               ("seeded topics", [r["coherence_seeded"] for r in records], "C0"),
               ("unseeded topics", [r["coherence_unseeded"] for r in records], "C2"),
           ] if seeded_topic_ids and unseeded_topic_ids else None)

    if use_diversity:
        _panel(axes[0, 1], [r["diversity"] for r in records],
               f"topic diversity (top-{diversity_top_n})",
               "Diversity vs. alpha\n(Dieng et al. 2020)", color="mediumpurple")
    else:
        axes[0, 1].axis("off")

    _panel(axes[1, 0], [r["doc_topic_entropy"] for r in records],
           "mean normalised entropy H(theta)/log(K)",
           "Document-topic entropy vs. alpha\n(1=diffuse mixtures, 0=single-topic docs)",
           color="teal")

    _panel(axes[1, 1], [r["doc_top_topic_share"] for r in records],
           "mean max_k(theta)",
           "Dominant-topic share vs. alpha\n(how much one topic dominates each doc)",
           color="indianred")

    if optimization_active:
        ax = axes[0, 2]
        final_means = [r["final_alpha_mean"] for r in records]
        final_stds = [r["final_alpha_std"] for r in records]
        ax.errorbar(alpha_list, final_means, yerr=final_stds, fmt="o", color="slategray",
                    ecolor="lightgray", capsize=3, label="post-training alpha (mean +- std over topics)")
        lo = min(min(alpha_list), min(final_means))
        hi = max(max(alpha_list), max(final_means))
        ax.plot([lo, hi], [lo, hi], linestyle=":", color="black", linewidth=1, label="y = x (no re-estimation)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("initial alpha (requested)"); ax.set_ylabel("post-training alpha (actual)")
        ax.set_title("Does tomotopy re-estimate alpha?\n(points near y=x => your alpha was respected;\nflat/clustered points => it was overridden)")
        ax.legend(fontsize=7)
        axes[1, 2].axis("off")

    fig.suptitle(f"LDA alpha sweep — {country_name} (K={k})", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"alpha_search_{country_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Alpha search plot saved to {plot_path}")

    results = {
        str(r["alpha"]): {
            "coherence": r["coherence"],
            "coherence_seeded": r["coherence_seeded"],
            "coherence_unseeded": r["coherence_unseeded"],
            "diversity": r["diversity"],
            "doc_topic_entropy": r["doc_topic_entropy"],
            "doc_top_topic_share": r["doc_top_topic_share"],
            "perplexity": r["perplexity"],
            **({"final_alpha_mean": r["final_alpha_mean"], "final_alpha_std": r["final_alpha_std"]}
               if optimization_active else {}),
        }
        for r in records
    }
    results["_meta"] = {
        "k": k,
        "best_alpha": best_alpha,
        "optim_interval": optim_interval,
        "hyperparameter_optimization_active": optimization_active,
    }
    results_path = os.path.join(output_dir, f"alpha_search_{country_name}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Alpha search results saved to {results_path}")

    return best_alpha, best_model, results


if __name__ == "__main__":
    print("This module provides run_alpha_search(). Import and call it from main.py "
          "(see the --alpha-search CLI flag) or a notebook.")
