import os
import numpy as np
import tomotopy as tp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from set_seeded_prior import set_seeded_prior


def _topic_diversity(model, top_n=25):
    """
    Fraction of unique words across the top-N words of every topic.

    TD = |union of top-N word sets| / (K * N)

    A value of 1.0 means no word appears in more than one topic's top-N list
    (maximum diversity). A value near 0 means topics are near-duplicates.

    Reference: Dieng, Ruiz & Blei (2020). "Topic Modeling in Embedding Spaces."
    Transactions of the Association for Computational Linguistics, 8, 439-453.
    """
    unique_words = set()
    for topic_id in range(model.k):
        top_words = [w for w, _ in model.get_topic_words(topic_id, top_n=top_n)]
        unique_words.update(top_words)
    return len(unique_words) / (model.k * top_n)


def find_best_k(documents, k_values, alpha=0.2, eta=0.001, min_cf=5,
                tw=None, n_iterations=5000, coherence_measure="c_v", top_n=20,
                use_diversity=True, diversity_top_n=25,
                w_coherence=0.5, w_diversity=0.3, w_perplexity=0.2,
                use_elbow=True,
                seed_lexicon=None, seed_weight=10.0, regular_weight=0.001,
                output_dir="output", country_name="all"):
    """
    Train LDA models over a range of K values and select the best K using a
    weighted combination of three complementary, scientifically validated metrics.

    Metrics and references
    ----------------------
    1. Perplexity  (lower is better)
       Statistical fit on training data.
       Blei, Ng & Jordan (2003). "Latent Dirichlet Allocation."
       Journal of Machine Learning Research, 3, 993-1022.

    2. c_v Coherence  (higher is better)
       Semantic interpretability of individual topics; shown to correlate best
       with human judgements across all coherence variants tested.
       Röder, Both & Hinneburg (2015). "Exploring the Space of Topic Coherence
       Measures." WSDM 2015, pp. 399-408.

    3. Topic Diversity  (higher is better, optional)
       Fraction of unique words in the union of each topic's top-N word list;
       penalises redundant near-duplicate topics that inflate perplexity and
       coherence without adding information.
       Dieng, Ruiz & Blei (2020). "Topic Modeling in Embedding Spaces."
       Transactions of the ACL, 8, 439-453.

    These three metrics are orthogonal: perplexity rewards more topics
    regardless of redundancy; coherence rewards within-topic word relatedness
    but ignores cross-topic overlap; diversity penalises that overlap.
    Combining all three is therefore better-motivated than any pair alone.

    The combined score normalises each metric to [0, 1] and computes:
        score = w_coherence  * norm_coherence
              + w_diversity  * norm_diversity   (if use_diversity=True)
              + w_perplexity * (1 - norm_perplexity)
    Weights are renormalised to sum to 1 if use_diversity=False.

    Parameters
    ----------
    use_diversity : bool
        Whether to include topic diversity. Default True.
    w_coherence, w_diversity, w_perplexity : float
        Relative weights (need not sum to 1; they are normalised internally).
        Defaults reflect that coherence is the most validated predictor of
        human interpretability (Röder et al. 2015).

    Returns
    -------
    best_k : int
        K that maximises the combined score.
    results : dict
        Maps each K to {"perplexity", "coherence", "diversity" (opt), "score"}.
    """
    if tw is None:
        tw = tp.TermWeight.IDF

    k_list = list(k_values)
    perplexities, coherences, diversities = [], [], []
    trained_models = {}

    active_metrics = ["coherence", "perplexity"] + (["diversity"] if use_diversity else [])
    print(f"\nSearching for best K over {k_list} "
          f"({n_iterations} iterations each, metrics={active_metrics})...")

    topic_name_to_id = (
        {name: i for i, name in enumerate(seed_lexicon.keys())}
        if seed_lexicon else {}
    )

    for k_val in tqdm(k_list, desc="K search"):
        model_k = tp.LDAModel(k=k_val, alpha=alpha, eta=eta, min_cf=min_cf, tw=tw)
        for doc in documents:
            model_k.add_doc(doc)
        if seed_lexicon:
            set_seeded_prior(model_k, seed_lexicon, topic_name_to_id,
                             seed_weight=seed_weight, regular_weight=regular_weight,
                             verbose=False)
        model_k.train(n_iterations)

        perp = model_k.perplexity
        coh_eval = tp.coherence.Coherence(model_k, coherence=coherence_measure, top_n=top_n)
        avg_coh = float(np.mean([coh_eval.get_score(topic_id=i) for i in range(model_k.k)]))
        div = _topic_diversity(model_k, top_n=diversity_top_n) if use_diversity else None

        perplexities.append(perp)
        coherences.append(avg_coh)
        diversities.append(div)
        trained_models[k_val] = model_k

        div_str = f"  diversity={div:.4f}" if use_diversity else ""
        tqdm.write(f"  K={k_val:>4d}  perplexity={perp:.4f}  coherence={avg_coh:.4f}{div_str}")

    # --- normalise to [0, 1] ------------------------------------------------
    def _norm(arr):
        a = np.array(arr, dtype=float)
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.full_like(a, 0.5)

    norm_perp = _norm(perplexities)
    norm_coh  = _norm(coherences)

    # renormalise weights to sum to 1
    if use_diversity:
        norm_div = _norm(diversities)
        total_w  = w_coherence + w_diversity + w_perplexity
        scores   = (w_coherence  * norm_coh
                    + w_diversity  * norm_div
                    + w_perplexity * (1.0 - norm_perp)) / total_w
    else:
        total_w = w_coherence + w_perplexity
        scores  = (w_coherence * norm_coh + w_perplexity * (1.0 - norm_perp)) / total_w

    def _elbow_idx(arr):
        """Perpendicular-distance elbow: index of maximum curvature in arr."""
        n = len(arr)
        if n <= 2:
            return 0
        x = np.linspace(0.0, 1.0, n)
        y = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        dists = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / (np.sqrt(dx**2 + dy**2) + 1e-10)
        return int(np.argmax(dists))

    max_idx     = int(np.argmax(scores))
    max_k       = k_list[max_idx]
    elbow_idx   = _elbow_idx(scores)
    elbow_k     = k_list[elbow_idx]
    best_k      = elbow_k if use_elbow else max_k
    best_k_perp = k_list[int(np.argmin(perplexities))]
    best_k_coh  = k_list[int(np.argmax(coherences))]

    print(f"\nK selection summary:")
    print(f"  Best K (elbow of combined score) : {elbow_k}  (score={scores[elbow_idx]:.4f})")
    print(f"  Best K (max combined score)      : {max_k}  (score={scores[max_idx]:.4f})")
    print(f"  Selected K                       : {best_k}  ({'elbow' if use_elbow else 'max'})")
    print(f"  Best K by coherence              : {best_k_coh}")
    print(f"  Best K by perplexity             : {best_k_perp}")
    if use_diversity:
        best_k_div = k_list[int(np.argmax(diversities))]
        print(f"  Best K by diversity              : {best_k_div}")

    # --- plot ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    n_panels = 4 if use_diversity else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    def _panel(ax, y, ylabel, title, color, alt_k=None, alt_label=None):
        ax.plot(k_list, y, marker="o", color=color)
        ax.axvline(best_k, color="red", linestyle="--", label=f"selected K={best_k}")
        if alt_k is not None and alt_k != best_k:
            ax.axvline(alt_k, color="orange", linestyle=":", linewidth=1.5, label=alt_label or f"K={alt_k}")
        ax.set_xlabel("K"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend()

    _panel(axes[0], perplexities, "Perplexity",
           "Perplexity vs. K\n(Blei et al. 2003)", "steelblue")
    _panel(axes[1], coherences, f"{coherence_measure} Coherence",
           f"Coherence vs. K\n(Röder et al. 2015)", "darkorange")
    if use_diversity:
        _panel(axes[2], diversities, f"Topic Diversity (top-{diversity_top_n})",
               "Diversity vs. K\n(Dieng et al. 2020)", "mediumpurple")
    _panel(axes[-1], scores, "Combined Score",
           f"Combined Score vs. K\n(w_coh={w_coherence}, w_div={w_diversity if use_diversity else 'off'}, w_perp={w_perplexity})",
           "green",
           alt_k=max_k if use_elbow else elbow_k,
           alt_label=f"max K={max_k}" if use_elbow else f"elbow K={elbow_k}")

    fig.suptitle(f"LDA K selection — {country_name}", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"k_selection_{country_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"K selection plot saved to {plot_path}")

    results = {}
    for i, k_val in enumerate(k_list):
        entry = {
            "perplexity": perplexities[i],
            "coherence":  coherences[i],
            "score":      float(scores[i]),
        }
        if use_diversity:
            entry["diversity"] = diversities[i]
        results[k_val] = entry

    results["_meta"] = {"elbow_k": elbow_k, "max_k": max_k, "selected_k": best_k}
    best_model = trained_models.pop(best_k)
    trained_models.clear()
    return best_k, best_model, results