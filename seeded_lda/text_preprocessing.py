import os
import re
from langdetect import detect, LangDetectException
import spacy
import stanza
import pandas as pd
from gensim.models import Phrases
import tomotopy as tp
from tqdm import tqdm
import pickle



def clean_scraped_text(text):
    # 1. Remove non-ASCII characters (This destroys the  artifacts)
    # text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # 2. Remove URLs and HTML tags (catches the >)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text) # HTML tags
    text = re.sub(r'\[.*?\]', '', text) # Brackets
    
    # 3. Remove emails and social media handles (catches @msnbc)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove weird artifacts and fix spacing
    text = text.replace('=', ' ')
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def is_year(text):
    return bool(re.match(r"^(18|19|20)\d{2}$", text))



def detect_languages_in_texts(df, text_col='Full_Text'):
    # 1. Initialize tqdm for pandas operations
    tqdm.pandas(desc="Detecting Languages")
    
    # 2. Swap .apply() with .progress_apply()
    df['Language'] = df[text_col].progress_apply(
        lambda x: detect(" ".join(x.split()[:25])) if isinstance(x, str) else 'unknown'
    )
    return df


def spacy_preprocessing_pipeline(df_w_texts, dict_name="en_core_web_sm", custom_words_to_remove=None):
    nlp = spacy.load(dict_name, disable=['parser'])

    if custom_words_to_remove is None:
        custom_words_to_remove = []

    for word in custom_words_to_remove:
        nlp.vocab[word].is_stop = True
        nlp.vocab[word.lower()].is_stop = True
        nlp.vocab[word.title()].is_stop = True
        nlp.vocab[word.upper()].is_stop = True

    cleaned_raw_texts = [clean_scraped_text(text) for text in df_w_texts['Full_Text'].tolist()]
    
    english_processed_docs = []

    for doc in tqdm(nlp.pipe(cleaned_raw_texts), disable=['parser'], total=len(cleaned_raw_texts), desc="Processing Texts"):
        clean_tokens = []
        for token in doc:
            # 1. Text Type Logic (Allows alphabetic words OR your specific year logic)
            is_valid_text = token.is_alpha or is_year(token.text)
            
            # 2. Filtering Logic (No stops, punct, or symbols)
            is_valid_type = not token.is_stop \
                            and not token.is_punct \
                            and token.pos_ != "SYM" \
                            and token.is_ascii \
                            # and token.pos_ in ['NOUN', 'ADJ']  # Keep only Nouns, Verbs, and Adjectives for better topic quality
            
            # 3. Entity Logic (Reverted to only drop PERSON,
            is_valid_ent = token.ent_type_ != "PERSON"
            
            if is_valid_text and is_valid_type and is_valid_ent:
                lemma = token.lemma_.lower().lstrip('_')
                if len(lemma) > 1 and lemma not in custom_words_to_remove:
                    clean_tokens.append(lemma)
                    
        english_processed_docs.append(clean_tokens)
        
    return english_processed_docs



def multiple_lang_preprocessing_pipeline(df_w_texts, custom_words_to_remove=None, remove_other_languages=True):

    # 1. SETUP BOTH LIBRARIES
    # Load spaCy for speed
    nlp_spacy = {
        'en': spacy.load("en_core_web_sm"),
        'ru': spacy.load("ru_core_news_sm"),
        'el': spacy.load("el_core_news_sm")
    }

    # Load Stanza for coverage (tokenize, lemma, and POS tagging are included)
    # We disable 'ner' (Named Entity Recognition) to make it run slightly faster
    nlp_stanza = {
        'ar': stanza.Pipeline('ar', processors='tokenize,mwt,pos,lemma', use_gpu=False),
        'tr': stanza.Pipeline('tr', processors='tokenize,pos,lemma', use_gpu=False)
    }

    def safe_detect(text):
        try:
            return detect(str(text))
        except LangDetectException:
            return "unknown"

    # 2. HYBRID PREPROCESSING LOOP
    print("Detecting languages and preprocessing...")
    native_tokenized_docs = [] 

    for idx, row in tqdm(df_w_texts.iterrows(), total=len(df_w_texts), desc="Cleaning Texts"):
        lang = safe_detect(row['Full_Text'])
        text = row['Full_Text']
        
        # Route 1: The Fast spaCy Path
        if lang in nlp_spacy:
            doc = nlp_spacy[lang](text)
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
            native_tokenized_docs.append((lang, tokens))
            
        # Route 2: The Accurate Stanza Fallback
        elif lang in nlp_stanza:
            doc = nlp_stanza[lang](text)
            tokens = []
            
            # Stanza structures data slightly differently (Doc -> Sentences -> Words)
            for sentence in doc.sentences:
                for word in sentence.words:
                    # Stanza doesn't have a built-in 'is_stop' flag like spaCy.
                    # But we can filter by Part of Speech (POS).
                    # We keep Nouns (NOUN), Verbs (VERB), and Adjectives (ADJ).
                    # This naturally strips out foreign stop words (pronouns, conjunctions, etc.)
                    if word.upos in ['NOUN', 'VERB', 'ADJ'] and word.lemma:
                        tokens.append(word.lemma.lower())
                        
            native_tokenized_docs.append((lang, tokens))
            
        # Route 3: True Unknowns (Junk data, or languages you completely ignore)
        else:
            native_tokenized_docs.append((lang, [])) 

    print("Preprocessing complete!")
    return native_tokenized_docs


def preprocess_texts(df_w_texts, custom_words_to_remove=None, remove_other_languages=True):
    
    if remove_other_languages:
        processed_docs = spacy_preprocessing_pipeline(df_w_texts, dict_name="en_core_web_sm", custom_words_to_remove=custom_words_to_remove)

    else:
        df_w_texts = detect_languages_in_texts(df_w_texts, text_col='Full_Text')
        processed_docs = multiple_lang_preprocessing_pipeline(df_w_texts, custom_words_to_remove=custom_words_to_remove, remove_other_languages=remove_other_languages)

    return processed_docs, df_w_texts


def n_gram_pipeline(english_processed_docs, min_count=15, threshold=0.005):

    # The N-Gramming Pipeline
    # Train the Gensim Phrases model on your cleaned text
    bigram_model = Phrases(english_processed_docs, min_count=min_count, threshold=threshold)

    # Apply the trained n-gram model back to your documents
    final_documents = []
    for doc in tqdm(english_processed_docs, desc="Applying N-Gram Model"):
        bigram_model[doc]  # This will transform the doc in-place, adding bigrams where detected
        final_documents.append(bigram_model[doc])

    return final_documents


def preprocess_pipeline(df_w_texts, custom_words_to_remove=None, remove_other_languages=True, output_dir="output", country_name=""):
    if not os.path.exists(f'processed_texts_{country_name}.pkl'):
        processed_docs, df_w_texts = preprocess_texts(df_w_texts, custom_words_to_remove=custom_words_to_remove, remove_other_languages=remove_other_languages)
        final_documents = n_gram_pipeline(processed_docs, min_count=15, threshold=0.005)

        # Save docs together with the dataframe indices so we can realign on load
        save_obj = {"docs": final_documents, "indices": df_w_texts.index.tolist()}
        with open(f'processed_texts_{country_name}.pkl', 'wb') as f:
            pickle.dump(save_obj, f)
    else:
        with open(f'processed_texts_{country_name}.pkl', 'rb') as f:
            saved = pickle.load(f)

        # If we saved the dict earlier, realign saved docs to current df_w_texts
        if isinstance(saved, dict) and "docs" in saved and "indices" in saved:
            saved_docs, saved_idx = saved["docs"], saved["indices"]
            aligned = [[] for _ in range(len(df_w_texts))]
            for doc, idx in zip(saved_docs, saved_idx):
                if 0 <= idx < len(aligned):
                    aligned[idx] = doc
            final_documents = aligned
        elif isinstance(saved, list):
            # Older format: list of docs. If lengths match, use directly, else pad/truncate to align with current df.
            if len(saved) == len(df_w_texts):
                final_documents = saved
            else:
                print(f"Warning: loaded processed_texts.pkl length={len(saved)} differs from current df length={len(df_w_texts)}. Aligning by padding/truncating.")
                aligned = [[] for _ in range(len(df_w_texts))]
                for i, doc in enumerate(saved):
                    if i < len(aligned):
                        aligned[i] = doc
                final_documents = aligned
        else:
            # Unknown format: create empty placeholders to match dataframe length
            print(f"Warning: processed_texts.pkl has unexpected format; creating empty placeholders of length {len(df_w_texts)}")
            final_documents = [[] for _ in range(len(df_w_texts))]

    return final_documents


def chunk_dataframe(df, token_col='Processed_Tokens', text_col='Full_Text', chunk_size=2000):
    """
    Slices long documents into smaller chunks while preserving the original text 
    and a reference to the original row index.
    """
    chunked_rows = []
    
    # Use tqdm to show a progress bar
    for original_idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        doc_tokens = row[token_col]
        original_text = row[text_col]
        
        # 1. Skip completely empty documents (Fixes the tomotopy empty doc bug!)
        if not isinstance(doc_tokens, list) or len(doc_tokens) == 0:
            continue
            
        # 2. If the document is shorter than the chunk size, keep it as one piece
        if len(doc_tokens) <= chunk_size:
            chunked_rows.append({
                'Original_Index': original_idx,
                'Chunk_ID': 1, # Just 1 chunk
                'Unified_Tokens': doc_tokens,
                'Full_Text': original_text
            })
            
        # 3. Slicing long documents
        else:
            chunk_counter = 1
            for i in range(0, len(doc_tokens), chunk_size):
                chunk = doc_tokens[i : i + chunk_size]
                
                # Only keep chunks that have enough words to be meaningful
                if len(chunk) > 20: 
                    chunked_rows.append({
                        'Original_Index': original_idx,
                        'Chunk_ID': chunk_counter,
                        'Unified_Tokens': chunk,
                        'Full_Text': original_text
                    })
                    chunk_counter += 1
                    
    # Return a brand new dataframe containing the split chunks
    return pd.DataFrame(chunked_rows)

