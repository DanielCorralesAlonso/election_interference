import pandas as pd
from tqdm import tqdm
import tomotopy as tp
import os

from webscrapping import webscrape_articles
from text_preprocessing import clean_scraped_text, chunk_dataframe, detect_languages_in_texts, preprocess_pipeline, collect_seed_term_candidates, report_seed_coverage
from config import custom_words_to_remove, seed_lexicon
from set_seeded_prior import set_seeded_prior
from utils import print_topic_overview, print_document_topics, print_corpus_topic_distribution, print_topic_distinctive_tokens, print_topic_coherence
from plot import plot_topic_evolution, plot_topic_evolution_comparison, plot_document_length_distribution
from topic_stability_analysis import run_topic_stability_pipeline
from find_best_k import find_best_k



if __name__ == "__main__":

    force_preprocess = "--force-preprocess" in os.sys.argv[1:]
    use_cached_df = "--use-cached-df" in os.sys.argv[1:]
    run_k_search = "--find-best-k" in os.sys.argv[1:]
    positional_args = [arg for arg in os.sys.argv[1:] if not arg.startswith("--")]
    country_name = positional_args[0] if len(positional_args) > 0 else "all"

    try:
        file_name = f"data/gdelt/GDELT_Extraction_2024_{country_name}_Election_Propaganda.csv"
        print(f"Attempting to load data from {file_name}...\n")
        df = pd.read_csv(file_name)
    except:
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


    # Drop texts that are too short (e.g., less than 100 characters) as they are unlikely to be informative for topic modeling.
    df_w_texts['Text_Length'] = df_w_texts['Full_Text'].apply(len)
    df_w_texts = df_w_texts[df_w_texts['Text_Length'] >= 100].reset_index(drop=True)
    print(f"After filtering out short texts (probably scrapping gone wrong), we have {len(df_w_texts)} articles.\n")
    
    for i in range(len(df_w_texts)):
        df_w_texts.loc[i, 'Full_Text'] = clean_scraped_text(df_w_texts.loc[i, 'Full_Text'])

    df_w_texts.drop(df_w_texts[df_w_texts['Full_Text'] == ''].index, inplace=True)
    print(f"After cleaning, we have {len(df_w_texts)} articles with non-empty text.\n")

    df_w_texts = detect_languages_in_texts(df_w_texts, text_col='Full_Text')
    print(f"Filtering to only English articles...")
    df_w_texts = df_w_texts[df_w_texts['Language'] == 'en'].reset_index(drop=True)
    print(f"After language filtering, we have {len(df_w_texts)} English articles.\n")
    for i in range(len(df_w_texts)):
        df_w_texts.loc[i, 'Full_Text'] = df_w_texts.loc[i, 'Full_Text'].encode('ascii', 'ignore').decode('utf-8')

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

    # Plot document length distribution before chunking
    plot_document_length_distribution(df_w_texts, text_col='Processed_Tokens', output_dir="output", country_name=country_name, title="Document Length Distribution Before Chunking")

    chunked_df = chunk_dataframe(df_w_texts, token_col='Processed_Tokens', text_col='Full_Text', chunk_size=2000)
    final_chunked_documents = chunked_df['Unified_Tokens'].tolist()
    plot_document_length_distribution(chunked_df, text_col='Unified_Tokens', output_dir="output", country_name=country_name, title="Document Length Distribution After Chunking")

    report_seed_coverage(
        final_chunked_documents,
        seed_lexicon,
        output_path=os.path.join("output", f"seed_coverage_{country_name}.json"),
    )

    alpha = 0.2
    eta = 0.001
    min_cf = 5
    tw = tp.TermWeight.IDF

    seed_weight = 10.0
    regular_weight = 0.001
    topic_name_to_id = {name: i for i, name in enumerate(seed_lexicon.keys())}
    topic_id_to_name = {i: name for name, i in topic_name_to_id.items()}

    ll_trace = []

    if run_k_search:
        k_range = range(50, 251, 25)
        k, model, k_search_results = find_best_k(
            final_chunked_documents,
            k_values=k_range,
            alpha=alpha,
            eta=eta,
            min_cf=min_cf,
            tw=tw,
            n_iterations=15000,
            coherence_measure="c_v",
            top_n=20,
            use_diversity=True,
            diversity_top_n=25,
            w_coherence=0.5,
            w_diversity=0.3,
            w_perplexity=0.2,
            seed_lexicon=seed_lexicon,
            seed_weight=seed_weight,
            regular_weight=regular_weight,
            output_dir="output",
            country_name=country_name,
        )
        print(f"\nReusing K={k} model from K search.")
    else:
        k = 150
        model = tp.LDAModel(k=k, alpha=alpha, eta=eta, min_cf=min_cf, tw=tw)
        print(f"Adding {len(final_chunked_documents)} documents to the model...")
        for doc in final_chunked_documents:
            model.add_doc(doc)
        model = set_seeded_prior(model, seed_lexicon, topic_name_to_id=topic_name_to_id,
                                 seed_weight=seed_weight, regular_weight=regular_weight)

        total_iterations = 10000
        burn_in = 7000
        sample_interval = 50
        print_interval = 500

        print(f"Training for {total_iterations} iterations...")
        model.train(0)

        with tqdm(total=total_iterations, desc="Gibbs Sampling") as pbar:
            for i in range(0, total_iterations, sample_interval):
                model.train(sample_interval)

                if (i + sample_interval) % print_interval == 0:
                    current_ll = model.ll_per_word
                    ll_trace.append(current_ll)
                    tqdm.write(f"Iteration: {i + sample_interval}\tLog-likelihood: {current_ll:.4f}")

                pbar.update(sample_interval)

        print("Training complete!")

    print(f"Model perplexity: {model.perplexity:.4f}")

        # 1. Build coherence evaluator bound to the model
    coherence_measure = "c_v"
    top_n = 30
    coh = tp.coherence.Coherence(model, coherence=coherence_measure, top_n=top_n)
    print_topic_coherence(model, coh, topic_id_to_name, output_dir="output", country_name=country_name, coherence_measure=coherence_measure, top_n=top_n)

    print_topic_overview(model, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)
    print_document_topics(model, df_w_texts, topic_id_to_name=topic_id_to_name, doc_index=0, output_dir="output", country_name=country_name)
    print_corpus_topic_distribution(model, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)
    print_topic_distinctive_tokens(model, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name, top_n=5)


    plot_topic_evolution(model, df_chunked=chunked_df, df_w_texts=df_w_texts, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)

    if country_name == "all":
        # Plot topic evolution for each country separately after the full corpus plot.
        for country in df_w_texts['Country'].unique():
            print(f"\nGenerating topic evolution plot for {country}...")
            country_df = df_w_texts[df_w_texts['Country'] == country]
            plot_topic_evolution(
                model,
                df_chunked=chunked_df,
                df_w_texts=country_df,
                topic_id_to_name=topic_id_to_name,
                output_dir="output",
                country_name=f"all_{country}",
            )

        plot_topic_evolution_comparison(
            model,
            df_chunked=chunked_df,
            df_w_texts=df_w_texts,
            topic_id_to_name=topic_id_to_name,
            output_dir="output",
            countries=("Russia", "China", "Iran"),
        )


    topic_stability_analysis = False
    if topic_stability_analysis:
        run_topic_stability_pipeline(final_chunked_documents, n_models=10, k=k, top_n=200, model_kwargs={"alpha": alpha, "eta": eta, "min_cf": min_cf, "tw": tw}, seeds=None, output_dir="output", reference_model=model, reference_name=country_name, seeded_topic_names=topic_id_to_name)

