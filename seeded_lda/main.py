import ast
import pandas as pd
from tqdm import tqdm
import tomotopy as tp
import numpy as np
import os
import random

from webscrapping import webscrape_articles
from text_preprocessing import clean_scraped_text, detect_languages_in_texts, preprocess_pipeline, collect_seed_term_candidates, report_seed_coverage
from config import custom_words_to_remove, seed_lexicon, election_dates
from set_seeded_prior import set_seeded_prior
from utils import print_topic_overview, print_document_topics, print_document_topics_by_country, print_corpus_topic_distribution, print_topic_coherence
from plot import plot_topic_evolution, plot_topic_evolution_comparison, plot_document_length_distribution, plot_document_entropy_by_country, plot_topic_cooccurrence, plot_narrative_by_country, plot_narrative_stacked_area, plot_all_topics_stacked_area
from topic_stability_analysis import run_topic_stability_pipeline
from find_best_k import find_best_k
from alpha_search import run_alpha_search



RANDOM_SEED = 40

if __name__ == "__main__":

    random.seed(RANDOM_SEED)

    force_preprocess = "--force-preprocess" in os.sys.argv[1:]
    use_cached_df = "--use-cached-df" in os.sys.argv[1:]
    skip_preprocess = "--skip-preprocess" in os.sys.argv[1:]
    run_k_search          = "--find-best-k"        in os.sys.argv[1:]
    alpha_values_arg      = next((a for a in os.sys.argv[1:] if a.startswith("--alpha-values")), None)
    alpha_search_values   = ([float(v) for v in alpha_values_arg.split("=")[1].split(",")]
                             if alpha_values_arg and "=" in alpha_values_arg
                             else None)
    optim_interval_arg    = next((a for a in os.sys.argv[1:] if a.startswith("--optim-interval")), None)
    # Alpha values provided → fix alpha (optim_interval=0) so the chosen value
    # is actually used. Explicit --optim-interval overrides this only when no
    # alpha values are given. Omitting both keeps tomotopy's default (None → 10).
    if alpha_search_values is not None:
        optim_interval = 0
    elif optim_interval_arg and "=" in optim_interval_arg:
        optim_interval = int(optim_interval_arg.split("=")[1])
    else:
        optim_interval = None
    run_stability         = "--stability-analysis" in os.sys.argv[1:]
    show_prominence       = "--prominence"         in os.sys.argv[1:]
    show_uncertainty      = "--uncertainty"        in os.sys.argv[1:]
    multi_seed_arg        = next((a for a in os.sys.argv[1:] if a.startswith("--multi-seed")), None)
    n_seeds               = int(multi_seed_arg.split("=")[1]) if multi_seed_arg and "=" in multi_seed_arg else 1
    posterior_samples_arg = next((a for a in os.sys.argv[1:] if a.startswith("--posterior-samples")), None)
    n_posterior_samples   = int(posterior_samples_arg.split("=")[1]) if posterior_samples_arg and "=" in posterior_samples_arg else 0
    train_workers_arg     = next((a for a in os.sys.argv[1:] if a.startswith("--train-workers")), None)
    n_train_workers       = int(train_workers_arg.split("=")[1]) if train_workers_arg and "=" in train_workers_arg else 0
    positional_args       = [a for a in os.sys.argv[1:] if not a.startswith("--")]
    country_name          = positional_args[0] if positional_args else "all"

    preprocess_cache_dir = os.path.join("output", "preprocess_cache")
    os.makedirs(preprocess_cache_dir, exist_ok=True)
    df_cache_path = os.path.join(preprocess_cache_dir, f"df_w_texts_{country_name}.parquet")

    if skip_preprocess and os.path.exists(df_cache_path):
        print(f"--skip-preprocess: loading cached dataframe from {preprocess_cache_dir}\n")
        df_w_texts = pd.read_parquet(df_cache_path)
        def _parse_tokens(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return ast.literal_eval(val)
            return []
        df_w_texts['Processed_Tokens'] = df_w_texts['Processed_Tokens'].apply(_parse_tokens)
        final_documents = df_w_texts['Processed_Tokens'].tolist()
        print(f"Loaded {len(df_w_texts)} articles from cache.\n")
    else:
        if skip_preprocess:
            print("--skip-preprocess requested but no cache found; running full pipeline.\n")

        try:
            file_name = f"data/gdelt/GDELT_Extraction_2024_{country_name}_Election_Propaganda.csv"
            print(f"Attempting to load data from {file_name}...\n")
            df = pd.read_csv(file_name)
        except FileNotFoundError:
            file_name = "data/gdelt/April16th_w_ELECTION_PROPAGANDA.csv"
            print(f"Failed to load country-specific file. Defaulting to {file_name}...\n")
            df = pd.read_csv(file_name)
            print(f"Loaded {len(df)} articles from the default file.\n")

        df.drop_duplicates(inplace=True)
        print(f"Loaded {len(df)} unique articles for country: {country_name}\n")

        df_w_texts = webscrape_articles(
                        df,
                        N=len(df),
                        random_state=42,
                        cache_file=f"data/gdelt/articles_with_texts_{country_name}.csv",
                        use_cached_df=use_cached_df,
                        save_cache=True
                    )

        df_w_texts.dropna(inplace=True)
        df_w_texts.reset_index(drop=True, inplace=True)
        print(f"After webscraping and dropping NaNs, we have {len(df_w_texts)} articles.\n")

        df_w_texts['Text_Length'] = df_w_texts['Full_Text'].apply(len)
        df_w_texts = df_w_texts[df_w_texts['Text_Length'] >= 100].reset_index(drop=True)
        print(f"After filtering out short texts (probably scrapping gone wrong), we have {len(df_w_texts)} articles.\n")

        df_w_texts['Full_Text'] = df_w_texts['Full_Text'].apply(clean_scraped_text)

        df_w_texts.drop(df_w_texts[df_w_texts['Full_Text'] == ''].index, inplace=True)
        print(f"After cleaning, we have {len(df_w_texts)} articles with non-empty text.\n")

        df_w_texts = detect_languages_in_texts(df_w_texts, text_col='Full_Text')
        print(f"Filtering to only English articles...")
        df_w_texts = df_w_texts[df_w_texts['Language'] == 'en'].reset_index(drop=True)
        print(f"After language filtering, we have {len(df_w_texts)} English articles.\n")
        df_w_texts['Full_Text'] = df_w_texts['Full_Text'].apply(lambda t: t.encode('ascii', 'ignore').decode('utf-8'))

        protected_seed_terms = collect_seed_term_candidates(seed_lexicon)
        preprocessing_result = preprocess_pipeline(
            df_w_texts,
            custom_words_to_remove=custom_words_to_remove,
            remove_other_languages=True,
            output_dir="output",
            country_name=country_name,
            force_preprocess=force_preprocess,
            seed_lexicon=seed_lexicon,
            min_df=2,
            max_df_ratio=0.9,
            bigram_min_count=15,
            bigram_threshold=0.005,
            protected_terms=protected_seed_terms,
            report_path=os.path.join("output", f"preprocessing_qa_{country_name}.json"),
        )

        # Canonical contract: preprocess_pipeline should return one list aligned to df_w_texts.
        # Some legacy variants return a tuple; keep only the document list and never overwrite df_w_texts here.
        if isinstance(preprocessing_result, tuple):
            print("Warning: legacy tuple returned by preprocess_pipeline; using only the documents component.")
            final_documents = preprocessing_result[0]
        else:
            final_documents = preprocessing_result

        if not isinstance(final_documents, list):
            final_documents = list(final_documents)

        if len(final_documents) != len(df_w_texts):
            raise ValueError(
                "preprocess_pipeline output is not aligned with input dataframe: "
                f"len(final_documents)={len(final_documents)} vs len(df_w_texts)={len(df_w_texts)}. "
                "This indicates an inconsistent preprocessing contract or stale implementation. "
                "Please ensure preprocess_pipeline returns exactly one document list aligned to the input rows."
            )

        df_w_texts['Processed_Tokens'] = final_documents

        plot_document_length_distribution(df_w_texts, text_col='Processed_Tokens', output_dir="output", country_name=country_name, title="Document Length Distribution")

        report_seed_coverage(
            final_documents,
            seed_lexicon,
            output_path=os.path.join("output", f"seed_coverage_{country_name}.json"),
        )

        # Save cache for --skip-preprocess on future runs
        df_cache = df_w_texts.copy()
        df_cache['Processed_Tokens'] = df_cache['Processed_Tokens'].apply(str)
        df_cache.to_parquet(df_cache_path, index=True)
        print(f"Preprocessing cache saved to {preprocess_cache_dir}\n")

    # Drop articles with too few tokens after preprocessing — these are scraping/
    # preprocessing artefacts, not real news articles.
    MIN_TOKENS = 5
    sufficient = [len(doc) >= MIN_TOKENS for doc in final_documents]
    n_dropped = len(final_documents) - sum(sufficient)
    if n_dropped:
        print(f"Dropping {n_dropped} document(s) with fewer than {MIN_TOKENS} tokens "
              f"(lengths: {sorted(len(d) for d, ok in zip(final_documents, sufficient) if not ok)}).")
    df_w_texts = df_w_texts[sufficient].reset_index(drop=True)
    final_documents = [doc for doc, keep in zip(final_documents, sufficient) if keep]
    print(f"After dropping short documents, we have {len(final_documents)} articles.\n")

    # --- corpus diagnostics -------------------------------------------------
    doc_lengths = [len(doc) for doc in final_documents]
    total_tokens = sum(doc_lengths)
    print(f"Corpus statistics:")
    print(f"  Documents      : {len(final_documents)}")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Tokens/doc     : mean={np.mean(doc_lengths):.0f}  median={np.median(doc_lengths):.0f}  "
          f"min={min(doc_lengths)}  max={max(doc_lengths)}")
    if 'Event_Date' in df_w_texts.columns:
        dates = pd.to_datetime(df_w_texts['Event_Date'].astype(str), format='%Y%m%d', errors='coerce').dropna()
        weekly_counts = dates.dt.to_period('W').value_counts()
        print(f"  Docs/week      : mean={weekly_counts.mean():.1f}  median={weekly_counts.median():.1f}  "
              f"min={weekly_counts.min()}  max={weekly_counts.max()}")
    if 'Country' in df_w_texts.columns:
        print(f"  Docs by country:\n" +
              "\n".join(f"    {c}: {n}" for c, n in df_w_texts['Country'].value_counts().items()))
    print()
    # -------------------------------------------------------------------------

    k = 30
    alpha = 0.000001
    eta = 0.01
    min_cf = 0  # Vocabulary already filtered by min_df in preprocessing (with seed protection); tomotopy adds no second filter.
    tw = tp.TermWeight.PMI

    top_n = 20

    seed_weight = 1.0
    regular_weight = 0.0
    topic_name_to_id = {name: i for i, name in enumerate(seed_lexicon.keys())}
    topic_id_to_name = {i: name for name, i in topic_name_to_id.items()}

    total_iterations = 8000

    # --- helper: train one model at fixed k ---------------------------------
    def _train_model(seed, verbose=True):
        m = tp.LDAModel(k=k, alpha=alpha, eta=eta, min_cf=min_cf, tw=tw, seed=seed)
        if optim_interval is not None:
            # tomotopy re-estimates alpha/eta from the data every
            # `optim_interval` iterations by default (10) — set to 0 so the
            # alpha/eta configured above are the values actually trained with.
            m.optim_interval = optim_interval
        for doc in final_documents:
            m.add_doc(doc)
        m = set_seeded_prior(m, seed_lexicon, topic_name_to_id=topic_name_to_id,
                             seed_weight=seed_weight, regular_weight=regular_weight,
                             verbose=verbose)
        m.train(0, workers=n_train_workers)
        if verbose:
            sample_interval, print_interval = 50, 500
            with tqdm(total=total_iterations, desc="Gibbs Sampling") as pbar:
                for i in range(0, total_iterations, sample_interval):
                    m.train(sample_interval, workers=n_train_workers)
                    if (i + sample_interval) % print_interval == 0:
                        tqdm.write(f"Iteration: {i + sample_interval}\tLog-likelihood: {m.ll_per_word:.4f}")
                    pbar.update(sample_interval)
        else:
            m.train(total_iterations, workers=n_train_workers)
        return m

    def _best_of_seeds(k_val, n, label=""):
        seeds = [RANDOM_SEED + i for i in range(n)]
        print(f"\nTraining {n} models (K={k_val}{label}), keeping best by C_V coherence...")
        seed_results = []

        def _train_and_score(seed):
            m = _train_model(seed, verbose=False)
            coh_eval = tp.coherence.Coherence(m, coherence="c_v", top_n=top_n)
            score = float(np.mean([coh_eval.get_score(topic_id=i) for i in range(m.k)]))
            return seed, m, score

        for seed in seeds:
            tqdm.write(f"  seed={seed} ...", end=" ")
            seed, m, score = _train_and_score(seed)
            seed_results.append((seed, m, score))
            tqdm.write(f"C_V = {score:.4f}")

        best_idx = int(np.argmax([r[2] for r in seed_results]))
        best_seed, best_model, best_score = seed_results[best_idx]
        print(f"Best seed: {best_seed}  (C_V = {best_score:.4f})")
        for seed, _, score in seed_results:
            marker = " <--" if seed == best_seed else ""
            print(f"  seed={seed}  coherence={score:.4f}{marker}")
        all_models = [r[1] for r in seed_results]
        other_models = [m for m in all_models if m is not best_model]
        return best_model, [best_model] + other_models

    # --- step 1: find K (optional) ------------------------------------------
    if run_k_search:
        k_range = range(20, 60, 5)
        k, model, k_search_results = find_best_k(
            final_documents,
            k_values=k_range,
            alpha=alpha,
            eta=eta,
            min_cf=min_cf,
            tw=tw,
            n_iterations=10000,
            coherence_measure="c_v",
            top_n=top_n,
            use_diversity=True,
            diversity_top_n=top_n,
            seed_lexicon=seed_lexicon,
            seed_weight=seed_weight,
            regular_weight=regular_weight,
            output_dir="output",
            country_name=country_name,
            random_seed=RANDOM_SEED,
            n_train_workers=n_train_workers,
            optim_interval=optim_interval,
        )
        print(f"\nK search selected K={k}.")
    else:
        k = k

    # --- step 1.5: alpha selection (optional) -----------------------------------
    # One alpha value  → use it directly (fixed, no search).
    # Multiple values  → run search over them and pick the best by C_V coherence.
    # No alpha values + optim_interval=0 → keep default alpha fixed.
    # No alpha values + optim_interval=None → tomotopy re-estimates alpha (default).
    if alpha_search_values is not None:
        if len(alpha_search_values) == 1:
            alpha = alpha_search_values[0]
            print(f"Single alpha value provided: using alpha={alpha:.2e} (fixed).")
        else:
            best_alpha, _, _ = run_alpha_search(
                final_documents,
                alpha_values=alpha_search_values,
                k=k,
                eta=eta,
                min_cf=min_cf,
                tw=tw,
                n_iterations=4000,
                coherence_measure="c_v",
                top_n=top_n,
                use_diversity=True,
                diversity_top_n=top_n,
                seed_lexicon=seed_lexicon,
                seed_weight=seed_weight,
                regular_weight=regular_weight,
                output_dir="output",
                country_name=country_name,
                random_seed=RANDOM_SEED,
                n_train_workers=n_train_workers,
                optim_interval=optim_interval,
            )
            print(f"Alpha search selected alpha={best_alpha:.2e} for main run.")
            alpha = best_alpha

    # --- step 2: train at chosen K (single seed or best-of-N) ---------------
    ensemble_models = None
    if n_seeds > 1:
        model, ensemble_models = _best_of_seeds(k, n_seeds, label=", from k-search" if run_k_search else "")
    elif not run_k_search:
        print(f"Adding {len(final_documents)} documents to the model...")
        model = _train_model(RANDOM_SEED, verbose=True)
    # else: k-search already produced a trained model — use it directly

    print(f"Model perplexity: {model.perplexity:.4f}")
    mdl = model

    # --- optional: thinned posterior sampling --------------------------------
    theta_samples = None
    if n_posterior_samples > 0:
        print(f"\nCollecting {n_posterior_samples} thinned posterior samples (200 iter/sample)...")
        theta_list = []
        for _ in tqdm(range(n_posterior_samples), desc="Posterior sampling"):
            model.train(200, workers=n_train_workers)
            theta_list.append(np.array([doc.get_topic_dist() for doc in model.docs]))
        theta_samples = np.stack(theta_list)  # (n_samples, n_docs, K)
        print(f"Posterior samples collected: shape {theta_samples.shape}")

    coherence_measure = "c_v"
    coh = tp.coherence.Coherence(model, coherence=coherence_measure, top_n=top_n)
    print_topic_coherence(mdl, coh, topic_id_to_name, output_dir="output", country_name=country_name, coherence_measure=coherence_measure, top_n=top_n)

    print_topic_overview(mdl, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)
    print_document_topics(mdl, df_w_texts, topic_id_to_name=topic_id_to_name, doc_index=0, output_dir="output", country_name=country_name, seed_lexicon=seed_lexicon)
    if 'Country' in df_w_texts.columns:
        print_document_topics_by_country(mdl, df_w_texts, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name, seed_lexicon=seed_lexicon)
    print_corpus_topic_distribution(mdl, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)

    plot_document_entropy_by_country(mdl, df_w_texts, output_dir="output", country_name=country_name)
    plot_topic_cooccurrence(mdl, output_dir="output", topic_id_to_name=topic_id_to_name, country_name=country_name)


    _country_topic_ids = {i for name, i in topic_name_to_id.items() if name in {"russia", "china", "iran"}}
    plot_topics = [i for i in sorted(topic_id_to_name.keys()) if i not in _country_topic_ids]
    country_actor_topics = sorted(_country_topic_ids)

    plot_topic_evolution(mdl, df_w_texts=df_w_texts, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name, topics_to_plot=plot_topics, theta_samples=theta_samples, show_prominence=show_prominence, show_uncertainty=show_uncertainty, election_dates=election_dates)

    if country_actor_topics:
        print("\nGenerating country-actor topic evolution plot...")
        plot_topic_evolution(mdl, df_w_texts=df_w_texts, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=f"{country_name}_actors", topics_to_plot=country_actor_topics, theta_samples=theta_samples, show_prominence=show_prominence, show_uncertainty=show_uncertainty, election_dates=election_dates)

    if country_name == "all":
        for country in df_w_texts['Country'].unique():
            print(f"\nGenerating topic evolution plot for {country}...")
            country_df = df_w_texts[df_w_texts['Country'] == country]
            plot_topic_evolution(
                mdl,
                df_w_texts=country_df,
                topic_id_to_name=topic_id_to_name,
                output_dir="output",
                country_name=f"all_{country}",
                topics_to_plot=plot_topics,
                theta_samples=theta_samples,
                show_prominence=show_prominence,
                show_uncertainty=show_uncertainty,
                election_dates=election_dates,
            )
            # Generate document topics using the common model, first doc from this country
            country_doc_idx = country_df.index[0]
            print_document_topics(mdl, df_w_texts, topic_id_to_name=topic_id_to_name,
                                  doc_index=country_doc_idx, output_dir="output",
                                  country_name=f"all_{country}", seed_lexicon=seed_lexicon)

        plot_topic_evolution_comparison(
            mdl,
            df_w_texts=df_w_texts,
            topic_id_to_name=topic_id_to_name,
            topics_to_plot=[i for i in plot_topics if i in topic_id_to_name],
            output_dir="output",
            countries=("Russia", "China", "Iran"),
            theta_samples=theta_samples,
            show_uncertainty=show_uncertainty,
            election_dates=election_dates,
        )

    if 'Country' in df_w_texts.columns:
        print("\nGenerating cross-country narrative comparison plot...")
        plot_narrative_by_country(
            mdl,
            df_w_texts=df_w_texts,
            topic_id_to_name=topic_id_to_name,
            narrative_topic_ids=plot_topics,
            output_dir="output",
            country_name=country_name,
            show_uncertainty=show_uncertainty,
            election_dates=election_dates,
        )

    print("\nGenerating stacked area narrative plot...")
    plot_narrative_stacked_area(
        mdl,
        df_w_texts=df_w_texts,
        topic_id_to_name=topic_id_to_name,
        narrative_topic_ids=plot_topics,
        output_dir="output",
        country_name=country_name,
        election_dates=election_dates,
    )

    print("\nGenerating full topic stacked area plot...")
    plot_all_topics_stacked_area(
        mdl,
        df_w_texts=df_w_texts,
        topic_id_to_name=topic_id_to_name,
        output_dir="output",
        country_name=country_name,
        election_dates=election_dates,
    )

    if run_stability:
        run_topic_stability_pipeline(
            final_documents,
            n_models=n_seeds if n_seeds > 1 else 10,
            k=k,
            top_n=20,
            model_kwargs={"alpha": alpha, "eta": eta, "min_cf": min_cf, "tw": tw},
            seeds=None,
            output_dir="output",
            reference_model=model,
            reference_name=country_name,
            seeded_topic_names=topic_id_to_name,
            ensemble_models=ensemble_models,
            n_train_workers=n_train_workers,
            optim_interval=optim_interval,
        )


