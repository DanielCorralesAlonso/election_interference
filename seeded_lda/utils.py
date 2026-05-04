
import os
import numpy as np
import pandas as pd

# ============================================================================
# HELPER FUNCTION: Escape LaTeX special characters
# ============================================================================
def escape_latex(text):
    """Escape special LaTeX characters in text"""
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    result = str(text)
    for char, replacement in special_chars.items():
        result = result.replace(char, replacement)
    return result


def write_latex_table(df, filepath, caption, label, column_format):
    """Write a pandas DataFrame as a LaTeX table with centering and small font."""
    styler = df.style.hide(axis="index")
    latex = styler.to_latex(
        column_format=column_format,
        hrules=True,
        label=label,
        caption=caption,
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        # Insert centering and small font directives
        latex = latex.replace('\\begin{table}', '\\begin{table}\n\\centering\n\\small')
        f.write(latex)


def _get_vocab_token(mdl, vocab_index):
    """Return the token string for a vocabulary index from tomotopy."""
    vocab_sources = [getattr(mdl, "used_vocabs", None), getattr(mdl, "vocabs", None)]
    for vocab_source in vocab_sources:
        if vocab_source is None:
            continue

        try:
            token = vocab_source[vocab_index]
            if isinstance(token, str):
                return token
            if hasattr(token, "surface"):
                return token.surface
            return str(token)
        except Exception:
            continue

    return str(vocab_index)


def print_topic_distinctive_tokens(mdl, topic_id_to_name, output_dir="output", country_name="", top_n=5):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"topic_distinctive_tokens_{country_name}.txt")
    latex_filepath = os.path.join(output_dir, f"topic_distinctive_tokens_{country_name}.tex")

    topic_counts = mdl.get_count_by_topics()
    total_words = float(sum(topic_counts))
    topic_priors = [count / total_words if total_words > 0 else 0.0 for count in topic_counts]

    topic_word_dists = [mdl.get_topic_word_dist(k_id) for k_id in range(mdl.k)]
    topic_word_matrix = pd.DataFrame(topic_word_dists)
    weighted_topic_matrix = topic_word_matrix.mul(topic_priors, axis=0)
    denominator = weighted_topic_matrix.sum(axis=0)

    text_rows = []
    latex_rows = []

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("========== TOPIC DISTINCTIVE TOKENS ==========\n\n")

        for topic_id in range(mdl.k):
            if topic_id in topic_id_to_name:
                topic_label = f"✅ SEEDED [{topic_id_to_name[topic_id]}]"
                latex_label = f"Topic [{escape_latex(topic_id_to_name[topic_id])}]"
            else:
                topic_label = f"⬜ UNSEEDED (Topic {topic_id})"
                latex_label = f"Topic {topic_id}"

            topic_probabilities = topic_word_matrix.iloc[topic_id]
            distinctiveness_scores = (topic_probabilities * topic_priors[topic_id]) / denominator
            distinctiveness_scores = distinctiveness_scores.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

            top_indices = distinctiveness_scores.sort_values(ascending=False).head(top_n).index.tolist()
            top_tokens = [
                (_get_vocab_token(mdl, vocab_index), float(distinctiveness_scores.iloc[vocab_index]))
                for vocab_index in top_indices
            ]

            f.write(f"{topic_label}\n")
            for token, score in top_tokens:
                f.write(f"   {token}: {score:.6f}\n")
            f.write("\n")

            text_rows.append({
                "Topic": topic_label,
                "Top distinctive tokens": ", ".join(f"{token} ({score:.4f})" for token, score in top_tokens),
            })

            latex_rows.append({
                "Topic": latex_label,
                "Top distinctive tokens": escape_latex(", ".join(f"{token} ({score:.4f})" for token, score in top_tokens)),
            })

    latex_df = pd.DataFrame(latex_rows, columns=["Topic", "Top distinctive tokens"])
    write_latex_table(
        latex_df,
        latex_filepath,
        caption="Topic Distinctive Tokens" + (f" - {country_name}" if country_name else ""),
        label="tab:topic_distinctive_tokens" + (f"_{country_name}" if country_name else ""),
        column_format="lp{12cm}",
    )


# ============================================================================
# 1. Get the total number of words assigned to each topic
# This helps you see how "big" or "heavy" each topic is in your corpus
# ============================================================================

def print_topic_overview(mdl, topic_id_to_name, output_dir="output", country_name=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"topic_overview_{country_name}.txt")
    latex_filepath = os.path.join(output_dir, f"topic_overview_{country_name}.tex")
    
    topic_counts = mdl.get_count_by_topics()

    # Prepare LaTeX table rows
    latex_rows = []

    with open(filepath, 'w', encoding = "utf-8") as f:
        f.write("\n========== ALL TOPICS OVERVIEW ==========\n\n")

        # Loop through every single topic from 0 up to K
        for k_id in range(mdl.k):
            
            # 2. Determine the Label (Seeded vs. Unseeded)
            if k_id in topic_id_to_name:
                # It's one of your seeded topics
                topic_label = f"✅ SEEDED [{topic_id_to_name[k_id]}]"
                latex_label = f"Topic [{escape_latex(topic_id_to_name[k_id])}]"
            else:
                # It's an unseeded noise topic
                topic_label = f"⬜ UNSEEDED (Topic {k_id})"
                latex_label = f"Topic {k_id}"
                
            # 3. Get the Top 10 Words
            top_words = [word for word, prob in mdl.get_topic_words(k_id, top_n=10)]
            latex_keywords = escape_latex(', '.join(top_words))
            
            # 4. Write the results clearly, including the word count
            f.write(f"{topic_label} | Size: {topic_counts[k_id]} words\n")
            f.write(f"   Keywords: {', '.join(top_words)}\n\n")
            
            # 5. Add LaTeX row data
            latex_rows.append({
                "Topic": latex_label,
                "Size (words)": topic_counts[k_id],
                "Keywords": latex_keywords,
            })

    latex_df = pd.DataFrame(latex_rows, columns=["Topic", "Size (words)", "Keywords"])
    write_latex_table(
        latex_df,
        latex_filepath,
        caption="Topic Overview" + (f" - {country_name}" if country_name else ""),
        label="tab:topic_overview" + (f"_{country_name}" if country_name else ""),
        column_format="lrp{8cm}",
    )






def print_document_topics(mdl, df_w_texts, topic_id_to_name, doc_index=0, output_dir="output", country_name=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"document_topics_{doc_index}_{country_name}.txt")
    latex_filepath = os.path.join(output_dir, f"document_topics_{doc_index}_{country_name}.tex")

    # 1. Select the document you want to inspect (e.g., Document index 42)
    target_doc = list(mdl.docs)[doc_index]

    # 2. Get the mathematical distribution
    # This returns a list of probabilities (one for each of your K topics) that sum to 1.0
    topic_dist = target_doc.get_topic_dist()

    # Prepare LaTeX table rows
    latex_rows = []

    with open(filepath, 'w', encoding = "utf-8") as f:
        f.write(f"--- Topic Distribution for Document #{doc_index} ---\n\n")

        # Optional: Write the original text so you can see if the math makes sense!
        # (Assuming your dataframe index matches the tomotopy document index)
        doc_text = df_w_texts['Full_Text'].iloc[doc_index][:200]
        f.write(f"Original Text snippet: {doc_text}...\n\n")

        # 3. Loop through the distribution and write the results clearly
        for topic_id, probability in enumerate(topic_dist):
            
            # We only want to write topics that actually have a meaningful presence in the document.
            # Let's ignore any topic that makes up less than 5% of the article.
            if probability > 0.05: 
                
                # Look up the human-readable name if it's a seeded topic
                topic_name = topic_id_to_name.get(topic_id, f"Unseeded Topic {topic_id}")
                
                # Convert the decimal probability to a clean percentage
                percentage = probability * 100
                
                # Write the Topic Name and its Percentage
                f.write(f"{topic_name}: {percentage:.1f}%\n")
                
                # Write the top 5 words of that topic so you know what it represents
                top_words = [word for word, prob in mdl.get_topic_words(topic_id, top_n=5)]
                f.write(f"   Keywords: {', '.join(top_words)}\n\n")
                
                # 4. Add LaTeX row data
                latex_rows.append({
                    "Topic Name": escape_latex(topic_name),
                    "Percentage": f"{percentage:.1f}\\%",
                    "Keywords": escape_latex(', '.join(top_words)),
                })

    latex_df = pd.DataFrame(latex_rows, columns=["Topic Name", "Percentage", "Keywords"])
    write_latex_table(
        latex_df,
        latex_filepath,
        caption=f"Document \\#{doc_index} Topic Distribution" + (f" - {country_name}" if country_name else ""),
        label="tab:document_topics_" + str(doc_index) + (f"_{country_name}" if country_name else ""),
        column_format="lrp{8cm}",
    )




def print_corpus_topic_distribution(mdl, topic_id_to_name, output_dir="output", country_name=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"corpus_topic_distribution_{country_name}.txt")
    latex_filepath = os.path.join(output_dir, f"corpus_topic_distribution_{country_name}.tex")

    # 1. Get the raw word counts for every topic
    topic_counts = mdl.get_count_by_topics()

    # 2. Calculate the total number of valid words the model processed
    total_words = sum(topic_counts)

    # Prepare LaTeX table rows
    latex_rows = []

    with open(filepath, 'w', encoding = "utf-8") as f:
        f.write(f"Total words modeled: {total_words:,}\n")
        f.write("\n========== CORPUS TOPIC DISTRIBUTION ==========\n\n")

        # 3. Calculate percentages and store them so we can sort them
        corpus_distribution = []

        for k_id in range(mdl.k):
            # Calculate the exact percentage of the corpus this topic occupies
            percentage = (topic_counts[k_id] / total_words) * 100 if total_words > 0 else 0
            
            # Apply the correct human-readable label
            if k_id in topic_id_to_name: 
                label = f"✅ SEEDED [{topic_id_to_name[k_id]}]"
                latex_label = f"Topic [{escape_latex(topic_id_to_name[k_id])}]"
            else:
                label = f"⬜ UNSEEDED (Topic {k_id})"
                latex_label = f"Topic {k_id}"
                
            # Append a tuple: (Percentage, Label, Raw Word Count, Topic ID, LaTeX Label)
            corpus_distribution.append((percentage, label, topic_counts[k_id], k_id, latex_label))

        # 4. Sort the list from the largest topic to the smallest topic
        corpus_distribution.sort(reverse=True, key=lambda x: x[0])

        # 5. Write the formatted results WITH Top Words
        for percentage, label, count, k_id, latex_label in corpus_distribution:
            # We only write topics that actually have words assigned to them
            if count > 0:
                # Extract the top 5 words for this specific topic
                top_words = [word for word, prob in mdl.get_topic_words(k_id, top_n=5)]
                
                # Write the stats and the keywords nicely formatted
                f.write(f"{label:<30} | {percentage:>5.2f}%  ({count:,} words)\n")
                f.write(f"   Keywords: {', '.join(top_words)}\n\n")
                
                # 6. Add LaTeX row data
                latex_rows.append({
                    "Topic": latex_label,
                    "Percentage": f"{percentage:.2f}\\%",
                    "Words": f"{count:,}",
                    "Keywords": escape_latex(', '.join(top_words)),
                })

    latex_df = pd.DataFrame(latex_rows, columns=["Topic", "Percentage", "Words", "Keywords"])
    if not latex_df.empty:
        latex_df.loc[len(latex_df)] = {
            "Topic": "",
            "Percentage": "\\textbf{Total:}",
            "Words": f"\\textbf{{{total_words:,}}}",
            "Keywords": "",
        }

    write_latex_table(
        latex_df,
        latex_filepath,
        caption="Corpus Topic Distribution" + (f" - {country_name}" if country_name else ""),
        label="tab:corpus_topic_dist" + (f"_{country_name}" if country_name else ""),
        column_format="lrrp{7cm}",
    )




def print_topic_coherence(mdl, coh, topic_id_to_name, output_dir="output", country_name="", coherence_measure="c_v", top_n=10):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"topic_coherence_{country_name}.txt")
    latex_filepath = os.path.join(output_dir, f"topic_coherence_{country_name}.tex")

    latex_rows = []

    with open(filepath, 'w', encoding="utf-8") as f:
        f.write(f"Coherence measure: {coherence_measure} | Top-N words: {top_n}\n")
        f.write("\n========== PER-TOPIC COHERENCE ==========\n\n")

        # 2. Compute per-topic coherence and collect results for sorting
        topic_coherence = []

        for k_id in range(mdl.k):
            score = coh.get_score(topic_id=k_id)

            if k_id in topic_id_to_name:
                label = f"✅ SEEDED [{topic_id_to_name[k_id]}]"
                latex_label = f"Topic [{escape_latex(topic_id_to_name[k_id])}]"
            else:
                label = f"⬜ UNSEEDED (Topic {k_id})"
                latex_label = f"Topic {k_id}"

            top_words = [word for word, prob in mdl.get_topic_words(k_id, top_n=5)]
            topic_coherence.append((score, label, top_words, k_id, latex_label))

        # 3. Sort from most to least coherent
        topic_coherence.sort(reverse=True, key=lambda x: x[0])

        # 4. Compute average across all topics
        all_scores = [score for score, *_ in topic_coherence]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # 5. Write results
        for score, label, top_words, k_id, latex_label in topic_coherence:
            f.write(f"{label:<30} | coherence: {score:>8.4f}\n")
            f.write(f"   Keywords: {', '.join(top_words)}\n\n")

            latex_rows.append({
                "Topic": latex_label,
                "Coherence": f"{score:.4f}",
                "Keywords": escape_latex(', '.join(top_words)),
            })

        f.write(f"{'Average coherence:':<30} | {avg_score:>8.4f}\n")

    # 6. Build and write LaTeX table
    latex_df = pd.DataFrame(latex_rows, columns=["Topic", "Coherence", "Keywords"])
    if not latex_df.empty:
        latex_df.loc[len(latex_df)] = {
            "Topic": "",
            "Coherence": f"\\textbf{{{avg_score:.4f}}}",
            "Keywords": "\\textbf{Average}",
        }

    write_latex_table(
        latex_df,
        latex_filepath,
        caption="Per-Topic Coherence" + (f" - {country_name}" if country_name else ""),
        label="tab:topic_coherence" + (f"_{country_name}" if country_name else ""),
        column_format="lrp{9cm}",
    )