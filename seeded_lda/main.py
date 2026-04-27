import pandas as pd
from tqdm import tqdm
import tomotopy as tp
import pdb
import os
import pickle
import numpy as np

from webscrapping import webscrape_articles
from text_preprocessing import clean_scraped_text, preprocess_texts, n_gram_pipeline, chunk_dataframe, detect_languages_in_texts, preprocess_pipeline
from config import custom_words_to_remove, seed_lexicon
from set_seeded_prior import set_seeded_prior
from utils import print_topic_overview, print_document_topics, print_corpus_topic_distribution
from plot import plot_topic_evolution
from topic_stability_analysis import run_topic_stability_pipeline
from topic_stability_analysis import run_topic_stability_pipeline

if __name__ == "__main__":

    country_name = os.sys.argv[1] if len(os.sys.argv) > 1 else "all"

    try:
        file_name = f"data/gdelt/GDELT_Extraction_2024_{country_name}_Election_Propaganda.csv"
        print(f"Attempting to load data from {file_name}...\n")
        df = pd.read_csv(file_name)
    except:
        file_name = "data/gdelt/April16th_w_ELECTION_PROPAGANDA.csv"
        print(f"Failed to load country-specific file. Defaulting to {file_name}...\n")
        df = pd.read_csv(file_name)

    df.drop_duplicates(inplace=True)

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

    
    for i in range(len(df_w_texts)):
        df_w_texts.loc[i, 'Full_Text'] = clean_scraped_text(df_w_texts.loc[i, 'Full_Text'])

    df_w_texts.drop(df_w_texts[df_w_texts['Full_Text'] == ''].index, inplace=True)

    # pdb.set_trace()
    df_w_texts = detect_languages_in_texts(df_w_texts, text_col='Full_Text')
    print(f"Filtering to only English articles...")
    df_w_texts = df_w_texts[df_w_texts['Language'] == 'en'].reset_index(drop=True)

    for i in range(len(df_w_texts)):
        df_w_texts.loc[i, 'Full_Text'] = df_w_texts.loc[i, 'Full_Text'].encode('ascii', 'ignore').decode('utf-8')

    # pdb.set_trace()
    final_documents = preprocess_pipeline(df_w_texts, custom_words_to_remove=custom_words_to_remove, remove_other_languages=True, output_dir="output", country_name=country_name)

    df_w_texts['Processed_Tokens'] = final_documents


    chunked_df = chunk_dataframe(df_w_texts, token_col='Processed_Tokens', text_col='Full_Text', chunk_size=2000)
    final_chunked_documents = chunked_df['Unified_Tokens'].tolist()


    k = 100
    alpha = 0.2
    eta = 0.001
    min_cf = 3
    tw = tp.TermWeight.IDF


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

    model = set_seeded_prior(model, seed_lexicon, topic_name_to_id=topic_name_to_id, seed_weight=4.0, regular_weight=0.001)

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

    print_topic_overview(model, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)
    print_document_topics(model, df_w_texts, topic_id_to_name=topic_id_to_name, doc_index=0, output_dir="output", country_name=country_name)
    print_corpus_topic_distribution(model, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)

    plot_topic_evolution(model, df_chunked=chunked_df, df_w_texts=df_w_texts, topic_id_to_name=topic_id_to_name, output_dir="output", country_name=country_name)

    topic_stability_analysis = False
    if topic_stability_analysis:
        run_topic_stability_pipeline(final_chunked_documents, n_models=5, k=k, top_n=15, model_kwargs={"alpha": alpha, "eta": eta, "min_cf": min_cf, "tw": tw}, seeds=None, output_dir="output", reference_model=model, reference_name=country_name, seeded_topic_names=topic_id_to_name)
