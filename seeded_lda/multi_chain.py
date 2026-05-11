import numpy as np
from scipy.optimize import linear_sum_assignment


class _DocWrapper:
    __slots__ = ("_dist",)

    def __init__(self, dist):
        self._dist = dist

    def get_topic_dist(self):
        return self._dist


class MultiChainSummary:
    """
    Drop-in replacement for a tomotopy LDAModel that exposes posteriors
    averaged across N independently initialised Gibbs chains.

    All K topics are Hungarian-aligned to the reference chain before
    averaging, so both seeded and unsupervised topics are correctly handled.
    """

    def __init__(self, reference_model, avg_theta, avg_phi):
        """
        Parameters
        ----------
        reference_model : tomotopy LDAModel  — used for vocabulary lookups
        avg_theta       : ndarray (n_docs, K) — aligned, averaged doc-topic dists
        avg_phi         : ndarray (K, vocab)  — aligned, averaged topic-word dists
        """
        self._ref = reference_model
        self._avg_theta = avg_theta
        self._avg_phi = avg_phi
        self.k = reference_model.k
        self.docs = [_DocWrapper(avg_theta[i]) for i in range(len(avg_theta))]

    def get_topic_words(self, topic_id, top_n=10):
        phi = self._avg_phi[topic_id]
        top_indices = np.argsort(phi)[::-1][:top_n]
        vocab = getattr(self._ref, "used_vocabs", None) or getattr(self._ref, "vocabs", None)
        if vocab is not None:
            return [(str(vocab[i]), float(phi[i])) for i in top_indices]
        return self._ref.get_topic_words(topic_id, top_n=top_n)

    def get_topic_word_dist(self, topic_id):
        return self._avg_phi[topic_id]

    def get_count_by_topics(self):
        # Approximate word counts from averaged theta (uniform doc-length weighting),
        # scaled to the same order of magnitude as the reference model's counts.
        counts = self._avg_theta.sum(axis=0)
        ref_total = float(sum(self._ref.get_count_by_topics()))
        scale = ref_total / (counts.sum() + 1e-10)
        return (counts * scale).tolist()

    @property
    def perplexity(self):
        return self._ref.perplexity

    @property
    def ll_per_word(self):
        return self._ref.ll_per_word

    @property
    def used_vocabs(self):
        return getattr(self._ref, "used_vocabs", None)

    @property
    def vocabs(self):
        return getattr(self._ref, "vocabs", None)

    def __getattr__(self, name):
        return getattr(self._ref, name)


def _align_chain(ref_phi, chain_phi, top_n=50):
    """
    Return the permutation array `perm` such that `chain_phi[perm]` is the
    best match for `ref_phi` under cosine similarity on the top-N words.

    Using only the top-N words removes the shared eta-smoothed baseline that
    inflates similarity scores when comparing on the full vocabulary, giving a
    cleaner signal for the Hungarian algorithm.

    cost_matrix[i, j] = cosine_distance(ref_topic_i, chain_topic_j)
    Hungarian minimises total cost → maximises total cosine similarity.
    """
    assert ref_phi.shape == chain_phi.shape, (
        f"phi shape mismatch: ref {ref_phi.shape} vs chain {chain_phi.shape}. "
        "All chains must be trained on the same documents with the same min_cf."
    )

    def _sparse_top_n(mat, n):
        """Zero out all but the top-n entries in each row."""
        out = np.zeros_like(mat)
        top_idx = np.argpartition(mat, -n, axis=1)[:, -n:]
        rows = np.arange(mat.shape[0])[:, None]
        out[rows, top_idx] = mat[rows, top_idx]
        return out

    def _unit_norm(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)  # dead topics → zero cosine sim
        return mat / norms

    ref_sparse   = _sparse_top_n(ref_phi,   top_n)
    chain_sparse = _sparse_top_n(chain_phi, top_n)

    sim_matrix  = _unit_norm(ref_sparse) @ _unit_norm(chain_sparse).T
    cost_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    _, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind, cost_matrix


def _alignment_report(cost_matrix, perm, topic_id_to_name, chain_idx):
    k = len(perm)
    seeded_ids = set(topic_id_to_name.keys())
    all_sims   = [1.0 - cost_matrix[i, perm[i]] for i in range(k)]
    seed_sims  = [1.0 - cost_matrix[i, perm[i]] for i in seeded_ids]

    print(f"    Chain {chain_idx + 1} alignment:")
    print(f"      mean cosine similarity — all topics:    {np.mean(all_sims):.4f}")
    print(f"      mean cosine similarity — seeded topics: {np.mean(seed_sims):.4f}")

    misaligned = [i for i in seeded_ids if perm[i] != i]
    if misaligned:
        print(f"      {len(misaligned)} seeded topic(s) found under a different slot in this chain (expected — alignment corrects this):")
        for i in misaligned:
            name = topic_id_to_name[i]
            sim = 1.0 - cost_matrix[i, perm[i]]
            print(f"        '{name}' (ref slot {i}) ← chain slot {perm[i]}  (cosine sim={sim:.3f})")
    else:
        print(f"      All seeded topics found in their designated slots.")


def build_multi_chain_summary(reference_model, final_documents, n_chains, base_seed,
                               k, alpha, eta, min_cf, tw,
                               seed_lexicon, seed_weight, regular_weight,
                               total_iterations, topic_name_to_id,
                               set_seeded_prior_fn):
    """
    Train (n_chains - 1) additional Gibbs chains, align every chain to the
    reference via the Hungarian algorithm, average aligned phi and theta, and
    return a MultiChainSummary.

    Reference chain (chain 0) is already trained; its topic ordering defines
    the canonical topic ID space.
    """
    import tomotopy as tp
    from tqdm import tqdm

    topic_id_to_name = {i: name for name, i in topic_name_to_id.items()}

    print(f"\nBuilding {n_chains}-chain posterior average with Hungarian alignment...")

    # --- chain 0: reference ---
    ref_phi   = np.array([reference_model.get_topic_word_dist(i) for i in range(k)])
    ref_theta = np.array([doc.get_topic_dist() for doc in reference_model.docs])

    all_phi   = [ref_phi]
    all_theta = [ref_theta]

    for chain_idx in range(1, n_chains):
        chain_seed = base_seed + chain_idx
        print(f"\n  Training chain {chain_idx + 1}/{n_chains} (seed={chain_seed})...")

        chain_model = tp.LDAModel(k=k, alpha=alpha, eta=eta, min_cf=min_cf, tw=tw, seed=chain_seed)
        for doc in final_documents:
            chain_model.add_doc(doc)
        chain_model = set_seeded_prior_fn(chain_model, seed_lexicon, topic_name_to_id=topic_name_to_id,
                                          seed_weight=seed_weight, regular_weight=regular_weight, verbose=False)

        with tqdm(total=total_iterations, desc=f"  Chain {chain_idx + 1}", leave=False) as pbar:
            interval = 500
            for _ in range(0, total_iterations, interval):
                chain_model.train(interval)
                pbar.update(interval)

        chain_phi   = np.array([chain_model.get_topic_word_dist(i) for i in range(k)])
        chain_theta = np.array([doc.get_topic_dist() for doc in chain_model.docs])

        perm, cost_matrix = _align_chain(ref_phi, chain_phi)
        _alignment_report(cost_matrix, perm, topic_id_to_name, chain_idx)

        all_phi.append(chain_phi[perm, :])
        all_theta.append(chain_theta[:, perm])

    avg_phi   = np.mean(all_phi,   axis=0)   # (K, vocab)
    avg_theta = np.mean(all_theta, axis=0)   # (n_docs, K)

    print(f"\nFinal averaged posterior from {n_chains} aligned chains.")
    return MultiChainSummary(reference_model, avg_theta, avg_phi)
