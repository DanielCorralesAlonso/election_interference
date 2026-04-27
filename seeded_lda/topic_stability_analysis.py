"""Topic stability analysis pipeline for LDA ensembles.

Approach:
- Use a reference model (e.g. the trained model from main.py) and an ensemble of additional models
- Extract top-N word sets for each topic
- Align topics from each model to the reference (Hungarian algorithm) using Jaccard similarity
- Compute stability scores per reference topic as the average pairwise Jaccard across aligned topics
- Produce plots and a human-readable summary. Seeded topics are highlighted when provided.
"""

from typing import List, Sequence, Tuple, Optional, Dict
import tomotopy as tp
import pickle
import os
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Added tqdm


def train_lda_models(
    docs: Sequence[Sequence[str]], 
    n_models: int = 5, 
    k: int = 100, 
    model_kwargs=None, 
    seeds: Sequence[int] = None
) -> List[tp.LDAModel]:
    """Train an ensemble of LDA models on the same corpus with different random seeds."""
    model_kwargs = model_kwargs or {}
    models = []
    
    if seeds is None:
        seeds = [i * 13 + 7 for i in range(n_models)]

    # OPTIMIZATION: Build the tomotopy corpus ONCE to save memory and time
    print("Building shared Tomotopy Corpus...")
    corpus = tp.utils.Corpus()
    for d in tqdm(docs, desc="Adding docs to corpus", leave=False):
        corpus.add_doc(d)

    # TQDM: Outer loop for the ensemble models
    for i in tqdm(range(n_models), desc="Training Ensemble Models"):
        seed = seeds[i]
        
        # BUG FIX: Actually pass the seed to the model! 
        # Passed the pre-built corpus instead of adding docs manually.
        m = tp.LDAModel(k=k, corpus=corpus, seed=seed, **model_kwargs)
        
        m.train(0) # Initialize
        
        # TQDM: Inner loop to track the 5000 training iterations
        train_iters = 5000
        chunk_size = 100
        with tqdm(total=train_iters, desc=f"Model {i+1}/{n_models} Iterations", leave=False) as pbar:
            for _ in range(0, train_iters, chunk_size):
                m.train(chunk_size)
                pbar.update(chunk_size)
                
        models.append(m)
        
    return models


def extract_top_word_sets_and_lists(models: Sequence[tp.LDAModel], top_n: int = 10) -> Tuple[List[List[set]], List[List[List[str]]]]:
    """Extract top-N word sets and ordered lists for each topic in every model."""
    all_models_sets = []
    all_models_lists = []
    
    print("Extracting top words from models...")
    for m in tqdm(models, desc="Extracting Topics", leave=False):
        topic_sets = []
        topic_lists = []
        for k_id in range(m.k):
            top_words = [w for w, prob in m.get_topic_words(k_id, top_n=top_n)]
            topic_sets.append(set(top_words))
            topic_lists.append(top_words)
        all_models_sets.append(topic_sets)
        all_models_lists.append(topic_lists)
        
    return all_models_sets, all_models_lists


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def align_models_to_reference(all_models_sets: List[List[set]]) -> List[List[set]]:
    """Align topics of all models to the first model (reference) using Hungarian algorithm."""
    if not all_models_sets:
        return []
    
    reference = all_models_sets[0]
    n_topics = len(reference)
    aligned = [reference]

    print("Aligning topics using Hungarian Algorithm...")
    for idx in tqdm(range(1, len(all_models_sets)), desc="Aligning Models", leave=False):
        other = all_models_sets[idx]
        sim = np.zeros((n_topics, n_topics), dtype=float)
        
        for i in range(n_topics):
            for j in range(n_topics):
                sim[i, j] = jaccard_similarity(reference[i], other[j])
                
        cost = 1.0 - sim
        row_ind, col_ind = linear_sum_assignment(cost)
        
        reordered = [set() for _ in range(n_topics)]
        for r, c in zip(row_ind, col_ind):
            reordered[r] = other[c]
        aligned.append(reordered)
        
    return aligned


def compute_stability_scores(aligned_models_sets: List[List[set]]) -> List[float]:
    if not aligned_models_sets:
        return []
    n_models = len(aligned_models_sets)
    n_topics = len(aligned_models_sets[0])
    scores = []
    
    for t in range(n_topics):
        sets = [aligned_models_sets[m][t] for m in range(n_models)]
        pair_sims = [jaccard_similarity(a, b) for a, b in combinations(sets, 2)]
        scores.append(float(sum(pair_sims) / len(pair_sims)) if pair_sims else 1.0)
        
    return scores


def run_topic_stability_pipeline(
    docs: Sequence[Sequence[str]],
    n_models: int = 5,
    k: int = 100,
    top_n: int = 10,
    model_kwargs=None,
    seeds: Sequence[int] = None,
    output_dir: str = "output",
    reference_model: Optional[tp.LDAModel] = None,
    reference_name: str = "reference",
    seeded_topic_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[float], dict]:
    """Run stability analysis using reference_model as the reference (if provided)."""
    os.makedirs(output_dir, exist_ok=True)
    model_kwargs = model_kwargs or {}

    models = []
    if reference_model is not None:
        models.append(reference_model)
        to_train = max(0, n_models - 1)
        if to_train > 0:
            print(f"Reference model provided. Training {to_train} additional ensemble models.")
            other_models = train_lda_models(docs, n_models=to_train, k=k, model_kwargs=model_kwargs, seeds=seeds)
            models.extend(other_models)
    else:
        print(f"No reference provided. Training {n_models} models from scratch (Model 0 will be reference).")
        models = train_lda_models(docs, n_models=n_models, k=k, model_kwargs=model_kwargs, seeds=seeds)

    # Extract sets and lists
    all_sets, all_lists = extract_top_word_sets_and_lists(models, top_n=top_n)

    # Align to reference (models[0])
    aligned_sets = align_models_to_reference(all_sets)

    # Compute scores
    scores = compute_stability_scores(aligned_sets)

    # Save results
    print(f"Saving results to {output_dir}...")
    results = {
        "n_models": len(models),
        "k": k,
        "top_n": top_n,
        "scores": scores,
        "aligned_sets": aligned_sets,
        "top_words_reference": all_lists[0] if all_lists else [],
    }
    with open(os.path.join(output_dir, f"topic_stability_results_{reference_name}.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Human-readable summary with seeded topic marking
    seeded_set = set(seeded_topic_names.keys()) if seeded_topic_names else set()
    with open(os.path.join(output_dir, f"topic_stability_summary_{reference_name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Topic stability summary (reference={reference_name}, n_models={len(models)}, k={k}, top_n={top_n})\n\n")
        for i, s in enumerate(scores):
            seed_label = f" SEEDED[{seeded_topic_names[i]}]" if i in seeded_set else ""
            top_words = ", ".join(all_lists[0][i]) if all_lists else ""
            f.write(f"Topic {i}:{seed_label} stability={s:.4f} | Top words: {top_words}\n")

    # Plots
    fig_path = os.path.join(output_dir, f"topic_stability_bar_{reference_name}.png")
    plt.figure(figsize=(14, 6))
    colors = ["C1" if i in seeded_set else "C0" for i in range(len(scores))]
    
    sns.barplot(
        x=list(range(len(scores))), 
        y=scores, 
        hue=list(range(len(scores))), 
        palette=colors, 
        legend=False                  
    )
    
    plt.xlabel("Topic index (reference)")
    plt.ylabel("Stability (avg pairwise Jaccard)")
    plt.title(f"Topic stability (reference={reference_name})")
    
    # FIX 1: Rotate x-axis ticks to prevent overlapping
    plt.xticks(rotation=90, fontsize=8)
    
    # FIX 2: Annotate ALL topics (Seeded names or Top-1 unseeded word)
    for i, s in enumerate(scores):
        if i in seeded_set:
            # Use brackets to visually distinguish seeded topics
            label_text = f"[{seeded_topic_names[i]}]" 
        else:
            # Fall back to the #1 most probable word in the reference model
            label_text = all_lists[0][i][0] if (all_lists and all_lists[0]) else ""
            
        plt.text(
            x=i, 
            y=s + 0.01, 
            s=label_text, 
            ha="center", 
            va="bottom", 
            fontsize=8, 
            rotation=90
        )
        
    # FIX 3: Pad the top of the Y-axis so the rotated text doesn't get cut off
    if scores:
        plt.ylim(0, max(scores) + 0.20)
        
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    # Heatmap
    n_topics = len(aligned_sets[0])
    avg_sim = np.zeros((n_topics, len(models) - 1)) if len(models) > 1 else np.zeros((n_topics, 1))
    for m_idx in range(1, len(aligned_sets)):
        for t in range(n_topics):
            avg_sim[t, m_idx - 1] = jaccard_similarity(aligned_sets[0][t], aligned_sets[m_idx][t])
            
    heatmap_path = os.path.join(output_dir, f"topic_alignment_heatmap_{reference_name}.png")
    plt.figure(figsize=(12, max(4, n_topics / 5)))
    sns.heatmap(avg_sim, cmap="viridis", cbar_kws={"label": "Jaccard similarity"})
    plt.xlabel("Model index (aligned to reference)")
    plt.ylabel("Reference topic index")
    plt.title("Jaccard similarity of reference topics to aligned topics per model")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    print("Pipeline complete.")
    return scores, results


if __name__ == "__main__":
    print("This module provides functions for topic stability analysis. Import and call run_topic_stability_pipeline().")