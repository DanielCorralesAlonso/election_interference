import os
import numpy as np
import tomotopy as tp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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
                seed_lexicon=None, seed_weight=10.0, regular_weight=0.001,
                output_dir="output", country_name="all", random_seed=None,
                n_workers=1, n_train_workers=0):
    """
    Train LDA models over a range of K values and select the best K by the
    elbow of the C_V coherence curve (Röder et al. 2015).

    Topic diversity (Dieng et al. 2020) is computed and shown as a diagnostic
    panel but does not affect K selection.

    Returns
    -------
    best_k : int
    best_model : trained LDAModel at best_k
    results : dict  — maps each K to {"coherence", "diversity" (opt)}
    """
    if tw is None:
        tw = tp.TermWeight.IDF

    k_list = list(k_values)
    coherences, diversities = [], []

    active_metrics = ["coherence"] + (["diversity"] if use_diversity else [])
    print(f"\nSearching for best K over {k_list} "
          f"({n_iterations} iterations each, metrics={active_metrics}, "
          f"{'parallel' if n_workers > 1 else 'sequential'}, workers={n_workers})...")

    topic_name_to_id = (
        {name: i for i, name in enumerate(seed_lexicon.keys())}
        if seed_lexicon else {}
    )

    def _train_k(k_val):
        m = tp.LDAModel(k=k_val, alpha=alpha, eta=eta, min_cf=min_cf, tw=tw, seed=random_seed)
        for doc in documents:
            m.add_doc(doc)
        if seed_lexicon:
            set_seeded_prior(m, seed_lexicon, topic_name_to_id,
                             seed_weight=seed_weight, regular_weight=regular_weight,
                             verbose=False)
        m.train(n_iterations, workers=n_train_workers)
        coh_eval = tp.coherence.Coherence(m, coherence=coherence_measure, top_n=top_n)
        avg_coh = float(np.mean([coh_eval.get_score(topic_id=i) for i in range(m.k)]))
        div = _topic_diversity(m, top_n=diversity_top_n) if use_diversity else None
        return k_val, avg_coh, div

    if n_workers > 1:
        k_results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_train_k, k_val): k_val for k_val in k_list}
            for fut in as_completed(futures):
                k_val, avg_coh, div = fut.result()
                k_results[k_val] = (avg_coh, div)
                div_str = f"  diversity={div:.4f}" if use_diversity else ""
                tqdm.write(f"  K={k_val:>4d}  coherence={avg_coh:.4f}{div_str}")
        for k_val in k_list:
            avg_coh, div = k_results[k_val]
            coherences.append(avg_coh)
            diversities.append(div)
    else:
        for k_val in tqdm(k_list, desc="K search"):
            k_val, avg_coh, div = _train_k(k_val)
            coherences.append(avg_coh)
            diversities.append(div)
            div_str = f"  diversity={div:.4f}" if use_diversity else ""
            tqdm.write(f"  K={k_val:>4d}  coherence={avg_coh:.4f}{div_str}")

    # --- select K by elbow of coherence curve --------------------------------
    coh_arr = np.array(coherences, dtype=float)

    def _elbow_idx(arr):
        n = len(arr)
        if n <= 2:
            return int(np.argmax(arr))
        x = np.linspace(0.0, 1.0, n)
        y = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        dists = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / (np.sqrt(dx**2 + dy**2) + 1e-10)
        return int(np.argmax(dists))

    elbow_idx = _elbow_idx(coh_arr)
    elbow_k   = k_list[elbow_idx]
    max_k     = k_list[int(np.argmax(coh_arr))]
    best_k    = elbow_k

    print(f"\nK selection summary (criterion: C_V coherence):")
    print(f"  Best K (elbow) : {elbow_k}  (coherence={coh_arr[elbow_idx]:.4f})")
    print(f"  Best K (max)   : {max_k}  (coherence={coh_arr[int(np.argmax(coh_arr))]:.4f})")
    print(f"  Selected K     : {best_k}")
    if use_diversity:
        print(f"  [diagnostic] Best K by diversity: {k_list[int(np.argmax(diversities))]}")

    # --- plot ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    n_panels = 2 if use_diversity else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    def _panel(ax, y, ylabel, title, color):
        ax.plot(k_list, y, marker="o", color=color)
        ax.axvline(best_k, color="red", linestyle="--", label=f"selected K={best_k}")
        if best_k != max_k:
            ax.axvline(max_k, color="orange", linestyle=":", linewidth=1.5, label=f"max K={max_k}")
        ax.set_xlabel("K"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend()

    _panel(axes[0], coherences, f"{coherence_measure} Coherence",
           f"Coherence vs. K\n(Röder et al. 2015)", "darkorange")
    if use_diversity:
        _panel(axes[1], diversities, f"Topic Diversity (top-{diversity_top_n})",
               "Diversity vs. K — diagnostic\n(Dieng et al. 2020)", "mediumpurple")

    fig.suptitle(f"LDA K selection — {country_name}", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"k_selection_{country_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"K selection plot saved to {plot_path}")

    results = {}
    for i, k_val in enumerate(k_list):
        entry = {"coherence": coherences[i]}
        if use_diversity:
            entry["diversity"] = diversities[i]
        results[k_val] = entry
    results["_meta"] = {"elbow_k": elbow_k, "max_k": max_k, "selected_k": best_k}

    print(f"Retraining model at selected K={best_k}...")
    best_model = _train_k(best_k)
    return best_k, best_model, results