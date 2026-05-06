import os
import json
import hashlib
import re
from langdetect import detect, LangDetectException
import spacy
import stanza
import pandas as pd
from collections import Counter, defaultdict
from gensim.models import Phrases
import tomotopy as tp
from tqdm import tqdm
import pickle
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
except Exception:
    # sklearn is optional for the more conservative deduplication method
    TfidfVectorizer = None
    NearestNeighbors = None
    np = None



def clean_scraped_text(text):
    if not isinstance(text, str):
        return ""

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
    
    return strip_boilerplate_text(text).strip()


def strip_boilerplate_text(text):
    if not isinstance(text, str):
        return ""

    boilerplate_patterns = [
        r"\bsubscribe\b.*",
        r"\bsign up\b.*",
        r"\bnewsletter\b.*",
        r"\bread more\b.*",
        r"\bclick here\b.*",
        r"\bprivacy policy\b.*",
        r"\bcookie policy\b.*",
        r"\ball rights reserved\b.*",
        r"\bterms of service\b.*",
        r"\bcontinue reading\b.*",
        r"\bopen in app\b.*",
        r"\bfollow us\b.*",
        r"\bshare this article\b.*",
        r"\bview comments\b.*",
        r"\brelated articles\b.*",
        r"\brecommended stories\b.*",
        r"\bsponsored content\b.*",
        r"\badvertisement\b.*",
        r"\bwatch live\b.*",
    ]

    stripped_text = text
    for pattern in boilerplate_patterns:
        stripped_text = re.sub(pattern, " ", stripped_text, flags=re.IGNORECASE | re.DOTALL)

    stripped_text = re.sub(r"\b(?:home|menu|search|login|register|share|follow|download|cookie|advertisement|newsletter)\b", " ", stripped_text, flags=re.IGNORECASE)
    stripped_text = re.sub(r"\s+", " ", stripped_text)
    return stripped_text.strip()


def is_year(text):
    return bool(re.match(r"^(18|19|20)\d{2}$", text))


def canonicalize_seed_term(term):
    if not isinstance(term, str):
        return ""

    normalized = term.strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def get_seed_term_candidates(term):
    if not isinstance(term, str):
        return []

    raw = term.strip().lower()
    candidates = [
        raw,
        raw.replace("-", "_"),
        raw.replace(" ", "_"),
        canonicalize_seed_term(term),
    ]

    if "_" in raw:
        candidates.append(raw.replace("_", ""))

    if "." in raw:
        candidates.append(raw.replace(".", ""))

    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def collect_seed_term_candidates(seed_lexicon):
    candidates = set()
    for words in seed_lexicon.values():
        for word in words:
            candidates.update(get_seed_term_candidates(word))
    return candidates


def collect_seed_phrase_patterns(seed_lexicon):
    patterns = []
    for words in seed_lexicon.values():
        for word in words:
            canonical_word = canonicalize_seed_term(word)
            parts = [part for part in re.split(r"[_\s]+", canonical_word) if part]
            if len(parts) >= 2:
                patterns.append(tuple(parts))

    unique_patterns = []
    seen_patterns = set()
    for pattern in sorted(patterns, key=len, reverse=True):
        if pattern not in seen_patterns:
            unique_patterns.append(pattern)
            seen_patterns.add(pattern)
    return unique_patterns


def normalize_text_for_dedup(text):
    if not isinstance(text, str):
        return ""

    normalized = clean_scraped_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_for_dedup(text):
    return [token for token in normalize_text_for_dedup(text).split() if len(token) >= 3]


def compute_simhash(tokens, hash_bits=64):
    if not tokens:
        return 0

    weights = Counter(tokens)
    accumulator = [0] * hash_bits

    for token, weight in weights.items():
        token_hash = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(hash_bits):
            bit_value = 1 << bit
            accumulator[bit] += weight if token_hash & bit_value else -weight

    fingerprint = 0
    for bit, value in enumerate(accumulator):
        if value > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(value_a, value_b):
    return (value_a ^ value_b).bit_count()


def simhash_band_keys(fingerprint, num_bands=2, band_size=32):
    mask = (1 << band_size) - 1
    return [((band_index), (fingerprint >> (band_index * band_size)) & mask) for band_index in range(num_bands)]


def remove_near_duplicates(df, text_col="Full_Text", similarity_threshold=5, min_tokens=50, num_bands=2, band_size=32, dedup_method="tfidf", tfidf_threshold=0.85):
    if df.empty:
        return df.copy(), {
            "input_documents": 0,
            "kept_documents": 0,
            "removed_documents": 0,
            "exact_duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "duplicate_groups": 0,
            "largest_cluster_size": 0,
            "similarity_threshold": similarity_threshold,
            "min_tokens": min_tokens,
        }

    kept_indices = []
    kept_rows = []
    kept_fingerprints = []
    kept_exact_signatures = {}
    band_buckets = defaultdict(list)
    duplicate_records = []
    exact_examples = []  # Store exact duplicate examples
    near_examples = []   # Store near duplicate examples

    # Two deduplication methods supported:
    #  - "simhash": original LSH-based simhash approach (default)
    #  - "tfidf": conservative TF-IDF + cosine similarity clustering (recommended when simhash is too noisy)
    if dedup_method == "tfidf":
        if TfidfVectorizer is None or NearestNeighbors is None:
            raise RuntimeError("scikit-learn is required for 'tfidf' deduplication. Install scikit-learn and retry.")

        texts = df[text_col].fillna("").tolist()
        token_counts = [len(tokenize_for_dedup(t)) for t in texts]
        # keep short documents (below min_tokens) out of TF-IDF deduplication
        eligible_idxs = [i for i, c in enumerate(token_counts) if c >= min_tokens]
        forced_keep_idxs = [i for i, c in enumerate(token_counts) if c < min_tokens]

        if not eligible_idxs:
            # nothing to dedupe with TF-IDF; keep all
            deduplicated_df = df.copy()
            exact_removed = 0
            near_removed = 0
            duplicate_records = []
        else:
            eligible_texts = [texts[i] for i in eligible_idxs]
            normalized_texts = [normalize_text_for_dedup(t) for t in eligible_texts]
            vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=1)
            X = vectorizer.fit_transform(normalized_texts)
            radius = 1.0 - float(tfidf_threshold)
            nbrs = NearestNeighbors(radius=radius, metric='cosine', algorithm='brute')
            nbrs.fit(X)
            distances, neighbors = nbrs.radius_neighbors(X, return_distance=True)

            # union-find to build connected components of similar docs
            n = len(eligible_idxs)
            parent = list(range(n))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for i, nbrs_idx in enumerate(neighbors):
                for j in nbrs_idx:
                    if i == j:
                        continue
                    union(i, j)

            groups = defaultdict(list)
            for i in range(n):
                groups[find(i)].append(i)

            dedup_keep_flags = [False] * len(df)
            # forced keep indices are always kept
            for fi in forced_keep_idxs:
                dedup_keep_flags[fi] = True

            # choose representative for each group and mark duplicates
            for comp in groups.values():
                comp_df_idxs = [eligible_idxs[i] for i in comp]
                # choose representative as the smallest original dataframe index (stable)
                rep_df_idx = int(min(comp_df_idxs))
                dedup_keep_flags[rep_df_idx] = True
                if len(comp_df_idxs) > 1:
                    # add sample for QA
                    rep_text = df.iloc[rep_df_idx][text_col][:250]
                    other_idx = comp_df_idxs[0] if comp_df_idxs[0] != rep_df_idx else comp_df_idxs[1]
                    other_text = df.iloc[other_idx][text_col][:250]
                    near_examples.append({
                        "type": "near",
                        "distance": None,
                        "representative_text": rep_text,
                        "duplicate_text": other_text,
                    })
                for idx in comp_df_idxs:
                    if idx == rep_df_idx:
                        continue
                    duplicate_records.append({
                        "duplicate_index": int(idx),
                        "representative_index": int(rep_df_idx),
                        "reason": "near",
                        "hamming_distance": None,
                    })

            # assemble deduplicated dataframe keeping order of first occurrence
            kept_positions = [i for i, flag in enumerate(dedup_keep_flags) if flag]
            kept_rows = [df.iloc[i] for i in kept_positions]
            deduplicated_df = pd.DataFrame(kept_rows).copy()
            if not deduplicated_df.empty:
                deduplicated_df.index = [int(df.index[i]) for i in kept_positions]

            exact_removed = sum(1 for r in duplicate_records if r.get("reason") == "exact")
            near_removed = sum(1 for r in duplicate_records if r.get("reason") == "near")

        # report will be created below
        report_extra = {"dedup_method": "tfidf", "tfidf_threshold": float(tfidf_threshold)}
    else:
        # fall back to original simhash-based approach
        report_extra = {"dedup_method": "simhash"}
        for original_index, row in df.iterrows():
            raw_text = row[text_col]
            normalized_text = normalize_text_for_dedup(raw_text)
            dedup_tokens = tokenize_for_dedup(raw_text)

            if len(dedup_tokens) < min_tokens:
                keep_row = True
                fingerprint = compute_simhash(dedup_tokens)
                exact_signature = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
            else:
                fingerprint = compute_simhash(dedup_tokens)
                exact_signature = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
                keep_row = True

            if exact_signature in kept_exact_signatures:
                representative_idx = kept_exact_signatures[exact_signature]
                representative_row = next((r for i, r in zip(kept_indices, kept_rows) if i == representative_idx), None)
                if representative_row is not None and len(exact_examples) < 3:
                    representative_text = representative_row[text_col][:250]
                    duplicate_text = raw_text[:250]
                    exact_examples.append({
                        "type": "exact",
                        "representative_text": representative_text,
                        "duplicate_text": duplicate_text,
                    })
                duplicate_records.append({
                    "duplicate_index": int(original_index),
                    "representative_index": int(kept_exact_signatures[exact_signature]),
                    "reason": "exact",
                })
                continue

            candidate_representatives = set()
            for band_key in simhash_band_keys(fingerprint, num_bands=num_bands, band_size=band_size):
                candidate_representatives.update(band_buckets[band_key])

            matched_representative = None
            matched_distance = None
            for representative_pos in candidate_representatives:
                representative_fingerprint = kept_fingerprints[representative_pos]
                distance = hamming_distance(fingerprint, representative_fingerprint)
                if distance <= similarity_threshold:
                    matched_representative = kept_indices[representative_pos]
                    matched_distance = distance
                    break

            if matched_representative is not None:
                representative_row = next((r for i, r in zip(kept_indices, kept_rows) if i == matched_representative), None)
                if representative_row is not None and len(near_examples) < 3:
                    representative_text = representative_row[text_col][:250]
                    duplicate_text = raw_text[:250]
                    near_examples.append({
                        "type": "near",
                        "distance": int(matched_distance),
                        "representative_text": representative_text,
                        "duplicate_text": duplicate_text,
                    })
                duplicate_records.append({
                    "duplicate_index": int(original_index),
                    "representative_index": int(matched_representative),
                    "reason": "near",
                    "hamming_distance": int(matched_distance),
                })
                continue

            kept_position = len(kept_indices)
            kept_indices.append(original_index)
            kept_rows.append(row)
            kept_fingerprints.append(fingerprint)
            kept_exact_signatures[exact_signature] = int(original_index)
            for band_key in simhash_band_keys(fingerprint, num_bands=num_bands, band_size=band_size):
                band_buckets[band_key].append(kept_position)

    # For simhash path we built `kept_rows` / `kept_indices` during the loop above.
    # For the TF-IDF path, `deduplicated_df` was already assembled and indexed inside that branch.
    if dedup_method != "tfidf":
        deduplicated_df = pd.DataFrame(kept_rows).copy()
        if not deduplicated_df.empty:
            deduplicated_df.index = kept_indices

    exact_removed = sum(1 for record in duplicate_records if record["reason"] == "exact")
    near_removed = sum(1 for record in duplicate_records if record["reason"] == "near")
    cluster_sizes = Counter(record["representative_index"] for record in duplicate_records)
    largest_cluster_size = max(cluster_sizes.values(), default=0) + 1 if cluster_sizes else 0
    representative_examples = []
    for representative_index, count in cluster_sizes.most_common(10):
        representative_examples.append({
            "representative_index": int(representative_index),
            "cluster_size": int(count + 1),
        })

    report = {
        "input_documents": int(len(df)),
        "kept_documents": int(len(deduplicated_df)),
        "removed_documents": int(len(duplicate_records)),
        "exact_duplicates_removed": int(exact_removed),
        "near_duplicates_removed": int(near_removed),
        "duplicate_groups": int(len(cluster_sizes)),
        "largest_cluster_size": int(largest_cluster_size),
        "similarity_threshold": int(similarity_threshold),
        "min_tokens": int(min_tokens),
        "num_bands": int(num_bands),
        "band_size": int(band_size),
        "duplicate_examples": representative_examples,
        "exact_duplicate_samples": exact_examples,
        "near_duplicate_samples": near_examples,
    }

    print(
        f"Near-duplicate removal: kept {report['kept_documents']} of {report['input_documents']} documents; "
        f"removed {report['removed_documents']} ({report['exact_duplicates_removed']} exact, {report['near_duplicates_removed']} near)."
    )
    if report["duplicate_groups"]:
        print(
            f"Duplicate clustering used simhash with threshold={similarity_threshold}; "
            f"largest cluster size={report['largest_cluster_size']}."
        )
    
    # Print textual examples
    all_examples = exact_examples + near_examples
    if all_examples:
        print(f"\nExample duplicate pairs (showing first {len(all_examples)}):")
        for i, example in enumerate(all_examples, 1):
            example_type = example["type"]
            if example_type == "exact":
                print(f"\n  [{i}] EXACT DUPLICATE:")
            else:
                distance = example.get("distance", "?")
                print(f"\n  [{i}] NEAR DUPLICATE (distance={distance}):")
            print(f"      Original: {example['representative_text'][:180]}...")
            print(f"      Removed:  {example['duplicate_text'][:180]}...")

    return deduplicated_df, report


def merge_seed_phrases_in_doc(tokens, phrase_patterns):
    if not tokens:
        return [], 0

    phrase_lookup = defaultdict(list)
    for pattern in phrase_patterns:
        phrase_lookup[pattern[0]].append(pattern)

    for first_token in phrase_lookup:
        phrase_lookup[first_token] = sorted(phrase_lookup[first_token], key=len, reverse=True)

    merged_tokens = []
    merge_count = 0
    index = 0
    normalized_tokens = [token.lower() for token in tokens]

    while index < len(tokens):
        first_token = normalized_tokens[index]
        matched_pattern = None

        for pattern in phrase_lookup.get(first_token, []):
            phrase_length = len(pattern)
            if normalized_tokens[index:index + phrase_length] == list(pattern):
                matched_pattern = pattern
                break

        if matched_pattern is not None:
            merged_tokens.append("_".join(matched_pattern))
            merge_count += 1
            index += len(matched_pattern)
        else:
            merged_tokens.append(tokens[index])
            index += 1

    return merged_tokens, merge_count


def merge_seed_phrases_in_documents(processed_docs, phrase_patterns):
    merged_documents = []
    total_merges = 0
    affected_documents = 0

    for doc in processed_docs:
        merged_doc, merge_count = merge_seed_phrases_in_doc(doc, phrase_patterns)
        if merge_count > 0:
            affected_documents += 1
            total_merges += merge_count
        merged_documents.append(merged_doc)

    return merged_documents, {"affected_documents": affected_documents, "total_merges": total_merges, "phrase_pattern_count": len(phrase_patterns)}


def build_preprocessing_config(
    custom_words_to_remove=None,
    remove_other_languages=True,
    min_df=2,
    max_df_ratio=0.9,
    bigram_min_count=15,
    bigram_threshold=0.005,
    protected_terms=None,
    seed_phrase_count=0,
    dedup_similarity_threshold=1,
    dedup_min_tokens=50,
    cache_version=3,
):
    return {
        "cache_version": cache_version,
        "remove_other_languages": remove_other_languages,
        "custom_words_to_remove": sorted(set(custom_words_to_remove or [])),
        "min_df": min_df,
        "max_df_ratio": max_df_ratio,
        "bigram_min_count": bigram_min_count,
        "bigram_threshold": bigram_threshold,
        "protected_terms": sorted(set(protected_terms or [])),
        "seed_phrase_count": seed_phrase_count,
        "dedup_similarity_threshold": dedup_similarity_threshold,
        "dedup_min_tokens": dedup_min_tokens,
    }


def hash_preprocessing_config(config):
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def filter_documents_by_document_frequency(processed_docs, min_df=2, max_df_ratio=0.9, protected_terms=None):
    protected_terms = set(protected_terms or [])
    if not processed_docs:
        return processed_docs, {"document_count": 0, "vocabulary_size": 0, "kept_vocabulary_size": 0}

    document_frequency = Counter()
    for doc in processed_docs:
        if isinstance(doc, list) and doc:
            document_frequency.update(set(doc))

    document_count = len(processed_docs)
    max_df = max(1, int(document_count * max_df_ratio))
    allowed_terms = {
        term
        for term, frequency in document_frequency.items()
        if term in protected_terms or (frequency >= min_df and frequency <= max_df)
    }

    filtered_docs = [[token for token in doc if token in allowed_terms] for doc in processed_docs]
    stats = {
        "document_count": document_count,
        "vocabulary_size": len(document_frequency),
        "kept_vocabulary_size": len(allowed_terms),
        "min_df": min_df,
        "max_df_ratio": max_df_ratio,
        "protected_terms": len(protected_terms),
    }
    return filtered_docs, stats


def report_seed_coverage(token_documents, seed_lexicon, output_path=None):
    vocabulary = set()
    for doc in token_documents:
        if isinstance(doc, list):
            vocabulary.update(token for token in doc if isinstance(token, str) and token)

    overall_total = 0
    overall_matched = 0
    topic_reports = {}
    missing_seed_terms = {}

    for topic_name, seed_terms in seed_lexicon.items():
        topic_total = len(seed_terms)
        topic_matched = 0
        matched_terms = {}
        missing_terms = []

        for term in seed_terms:
            overall_total += 1
            match = next((candidate for candidate in get_seed_term_candidates(term) if candidate in vocabulary), None)
            if match is None:
                missing_terms.append(term)
                continue

            overall_matched += 1
            topic_matched += 1
            matched_terms[term] = match

        topic_reports[topic_name] = {
            "total": topic_total,
            "matched": topic_matched,
            "coverage": topic_matched / topic_total if topic_total else 0.0,
            "matched_terms": matched_terms,
            "missing_terms": missing_terms,
        }
        if missing_terms:
            missing_seed_terms[topic_name] = missing_terms

    report = {
        "overall": {
            "total": overall_total,
            "matched": overall_matched,
            "coverage": overall_matched / overall_total if overall_total else 0.0,
            "vocabulary_size": len(vocabulary),
        },
        "topics": topic_reports,
        "missing_seed_terms": missing_seed_terms,
    }

    print(
        f"Seed coverage: {report['overall']['matched']}/{report['overall']['total']} "
        f"({report['overall']['coverage']:.1%}) across a vocabulary of {report['overall']['vocabulary_size']} tokens."
    )

    for topic_name, topic_report in topic_reports.items():
        print(
            f"  - {topic_name}: {topic_report['matched']}/{topic_report['total']} "
            f"({topic_report['coverage']:.1%})"
        )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    return report



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


def preprocess_pipeline(
    df_w_texts,
    custom_words_to_remove=None,
    remove_other_languages=True,
    output_dir="output",
    country_name="",
    force_preprocess=False,
    seed_lexicon=None,
    min_df=2,
    max_df_ratio=0.9,
    bigram_min_count=15,
    bigram_threshold=0.005,
    protected_terms=None,
    dedup_similarity_threshold=1,
    dedup_min_tokens=50,
    report_path=None,
    cache_version=3,
):
    os.makedirs(output_dir, exist_ok=True)
    original_length = len(df_w_texts)
    cache_path = os.path.join(output_dir, f"processed_texts_{country_name}.pkl")
    legacy_cache_path = f"processed_texts_{country_name}.pkl"
    phrase_patterns = collect_seed_phrase_patterns(seed_lexicon or {})
    preprocessing_config = build_preprocessing_config(
        custom_words_to_remove=custom_words_to_remove,
        remove_other_languages=remove_other_languages,
        min_df=min_df,
        max_df_ratio=max_df_ratio,
        bigram_min_count=bigram_min_count,
        bigram_threshold=bigram_threshold,
        protected_terms=protected_terms,
        seed_phrase_count=len(phrase_patterns),
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_min_tokens=dedup_min_tokens,
        cache_version=cache_version,
    )
    config_hash = hash_preprocessing_config(preprocessing_config)
    qa_report = {
        "config_hash": config_hash,
        "config": preprocessing_config,
        "cache": {"hit": False, "source": None},
    }

    def load_cache(path):
        if not os.path.exists(path):
            return None

        with open(path, "rb") as handle:
            return pickle.load(handle)

    saved = load_cache(cache_path)
    if saved is None:
        saved = load_cache(legacy_cache_path)

    cache_is_valid = False
    if force_preprocess:
        print("Force preprocessing enabled; ignoring any cached processed texts.")
        saved = None

    if isinstance(saved, dict):
        cache_is_valid = (
            saved.get("config_hash") == config_hash
            and "docs" in saved
            and "indices" in saved
            and len(saved["docs"]) == original_length
            and len(saved["indices"]) == original_length
        )
    if cache_is_valid:
        saved_docs, saved_idx = saved["docs"], saved["indices"]
        aligned = [[] for _ in range(len(df_w_texts))]
        for doc, idx in zip(saved_docs, saved_idx):
            if 0 <= idx < len(aligned):
                aligned[idx] = doc
        final_documents = aligned
        qa_report["cache"] = {"hit": True, "source": cache_path if os.path.exists(cache_path) else legacy_cache_path}
    else:
        deduplicated_df, dedup_report = remove_near_duplicates(
            df_w_texts,
            text_col="Full_Text",
            similarity_threshold=dedup_similarity_threshold,
            min_tokens=dedup_min_tokens,
        )
        qa_report["deduplication"] = dedup_report
        dedup_indices = deduplicated_df.index.tolist()

        processed_docs, df_w_texts = preprocess_texts(
            deduplicated_df,
            custom_words_to_remove=custom_words_to_remove,
            remove_other_languages=remove_other_languages,
        )

        merged_docs, phrase_report = merge_seed_phrases_in_documents(processed_docs, phrase_patterns)
        qa_report["seed_phrase_merging"] = phrase_report

        filtered_docs, _ = filter_documents_by_document_frequency(
            merged_docs,
            min_df=min_df,
            max_df_ratio=max_df_ratio,
            protected_terms=protected_terms,
        )
        qa_report["document_frequency_filtering"] = {
            "input_documents": len(merged_docs),
            "kept_documents": len(filtered_docs),
            "min_df": min_df,
            "max_df_ratio": max_df_ratio,
            "protected_terms": len(set(protected_terms or [])),
        }
        compact_documents = n_gram_pipeline(filtered_docs, min_count=bigram_min_count, threshold=bigram_threshold)
        final_documents = [[] for _ in range(original_length)]
        for doc, original_index in zip(compact_documents, dedup_indices):
            if 0 <= original_index < len(final_documents):
                final_documents[original_index] = doc

        save_obj = {
            "docs": final_documents,
            "indices": list(range(original_length)),
            "config_hash": config_hash,
            "preprocessing_config": preprocessing_config,
        }
        with open(cache_path, "wb") as handle:
            pickle.dump(save_obj, handle)

        qa_report["cache"] = {"hit": False, "source": cache_path}

    if report_path:
        qa_report["final_documents"] = len(final_documents)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(qa_report, handle, indent=2, sort_keys=True)

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

