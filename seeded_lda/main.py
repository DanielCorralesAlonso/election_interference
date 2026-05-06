import pandas as pd
from tqdm import tqdm
import tomotopy as tp
import pdb
import os
import pickle
import numpy as np

from webscrapping import webscrape_articles
from text_preprocessing import clean_scraped_text, preprocess_texts, n_gram_pipeline, chunk_dataframe, detect_languages_in_texts, preprocess_pipeline, collect_seed_term_candidates, report_seed_coverage, build_preprocessing_config
from config import custom_words_to_remove, seed_lexicon
from set_seeded_prior import set_seeded_prior
from utils import print_topic_overview, print_document_topics, print_corpus_topic_distribution, print_topic_distinctive_tokens, print_topic_coherence
from plot import plot_topic_evolution, plot_document_length_distribution
from topic_stability_analysis import run_topic_stability_pipeline

if __name__ == "__main__":

    force_preprocess = "--force-preprocess" in os.sys.argv[1:]
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
                    use_cached_df=True, 
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

    # pdb.set_trace()
    df_w_texts = detect_languages_in_texts(df_w_texts, text_col='Full_Text')
    print(f"Filtering to only English articles...")
    df_w_texts = df_w_texts[df_w_texts['Language'] == 'en'].reset_index(drop=True)
    print(f"After language filtering, we have {len(df_w_texts)} English articles.\n")
    for i in range(len(df_w_texts)):
        df_w_texts.loc[i, 'Full_Text'] = df_w_texts.loc[i, 'Full_Text'].encode('ascii', 'ignore').decode('utf-8')

    # pdb.set_trace()
    protected_seed_terms = collect_seed_term_candidates(seed_lexicon)
    final_documents = preprocess_pipeline(
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

    k = 100
    alpha = 0.2 
    eta = 0.001  # Will be changes to seeded prior.
    min_cf = 5
    tw = tp.TermWeight.IDF

    seed_weight = 10.0
    regular_weight = 0.001

    #pdb.set_trace()

    model = tp.LDAModel(
                k=k, 
                alpha=alpha, 
                eta=eta, 
                min_cf=min_cf, 
                # rm_top=100, 
                tw = tw
            )
    
    # 2. Add Documents
    print(f"Adding {len(final_chunked_documents)} documents to the model...")
    for doc in final_chunked_documents:
        model.add_doc(doc)

    topic_name_to_id = {name: i for i, name in enumerate(seed_lexicon.keys())}
    topic_id_to_name = {i: name for name, i in topic_name_to_id.items()}

    model = set_seeded_prior(model, seed_lexicon, topic_name_to_id=topic_name_to_id, seed_weight=seed_weight, regular_weight=regular_weight)


    total_iterations = 8000
    burn_in = 7000           # Wait until the model has converged to start sampling
    sample_interval = 50     # Take a snapshot every 50 iterations to avoid autocorrelation
    print_interval = 500     # Keep your standard logging interval

    print(f"Training for {total_iterations} iterations...")
    model.train(0) # Initialize parameters

    phi_samples = [] # List to store the topic-word distributions
    ll_trace = []    # List to store log-likelihood for plotting later

    with tqdm(total=total_iterations, desc="Gibbs Sampling") as pbar:
        for i in range(0, total_iterations, sample_interval):
            model.train(sample_interval)
            
            # 1. Log-Likelihood Tracking
            if (i + sample_interval) % print_interval == 0:
                current_ll = model.ll_per_word
                ll_trace.append(current_ll)
                # tqdm.write ensures the print doesn't break the progress bar visual
                tqdm.write(f"Iteration: {i + sample_interval}\tLog-likelihood: {current_ll:.4f}")
                
            # 2. MCMC Posterior Sampling (Only after burn-in!)
            if (i + sample_interval) > burn_in:
                # model.get_topic_word_dist(k) returns a 1D array of word probabilities for topic K
                # We build a 2D matrix (K topics x V vocabulary) for this specific sample
                current_phi = np.array([model.get_topic_word_dist(k) for k in range(model.k)])
                phi_samples.append(current_phi)
                
            pbar.update(sample_interval)

    print("Training complete!")

    # ==========================================
    # 6. Calculate Topic Uncertainty (Variance)
    # ==========================================
    '''# Convert list of 2D matrices into a 3D tensor: Shape (num_samples, K, V)
    phi_samples_tensor = np.array(phi_samples)

    # Calculate the variance across the samples (axis=0)
    # Resulting matrix shape: (K, V) -> The variance of every word in every topic
    phi_variance = np.var(phi_samples_tensor, axis=0)

    # Example: Get the average variance for each topic to see which topics are most unstable
    average_topic_variance = np.mean(phi_variance, axis=1)

    print("\n--- Topic Uncertainty Report ---")
    for k in range(model.k):
        # Differentiate your seeded topics vs unseeded based on your dictionaries
        topic_name = topic_id_to_name.get(k, f"Unseeded_{k}")
        print(f"Topic {k} ({topic_name}) Average Word Variance: {average_topic_variance[k]:.8f}")
'''

    print(f"Model perplexity: {model.perplexity:.4f}")

        # 1. Build coherence evaluator bound to the model
    coherence_measure = "c_v"
    top_n = 10
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


    topic_stability_analysis = False
    if topic_stability_analysis:
        run_topic_stability_pipeline(final_chunked_documents, n_models=10, k=k, top_n=100, model_kwargs={"alpha": alpha, "eta": eta, "min_cf": min_cf, "tw": tw}, seeds=None, output_dir="output", reference_model=model, reference_name=country_name, seeded_topic_names=topic_id_to_name)

