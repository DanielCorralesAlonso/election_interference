
import os

# 1. Get the total number of words assigned to each topic
# This helps you see how "big" or "heavy" each topic is in your corpus

def print_topic_overview(mdl, topic_id_to_name, output_dir="output", country_name=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"topic_overview_{country_name}.txt")
    
    topic_counts = mdl.get_count_by_topics()

    with open(filepath, 'w', encoding = "utf-8") as f:
        f.write("\n========== ALL TOPICS OVERVIEW ==========\n\n")

        # Loop through every single topic from 0 up to K
        for k_id in range(mdl.k):
            
            # 2. Determine the Label (Seeded vs. Unseeded)
            if k_id in topic_id_to_name:
                # It's one of your seeded topics
                topic_label = f"✅ SEEDED [{topic_id_to_name[k_id]}]"
            else:
                # It's an unseeded noise topic
                topic_label = f"⬜ UNSEEDED (Topic {k_id})"
                
            # 3. Get the Top 10 Words
            top_words = [word for word, prob in mdl.get_topic_words(k_id, top_n=10)]
            
            # 4. Write the results clearly, including the word count
            f.write(f"{topic_label} | Size: {topic_counts[k_id]} words\n")
            f.write(f"   Keywords: {', '.join(top_words)}\n\n")





def print_document_topics(mdl, df_w_texts, topic_id_to_name, doc_index=0, output_dir="output", country_name=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"document_topics_{doc_index}_{country_name}.txt")

    # 1. Select the document you want to inspect (e.g., Document index 42)
    target_doc = list(mdl.docs)[doc_index]

    # 2. Get the mathematical distribution
    # This returns a list of probabilities (one for each of your K topics) that sum to 1.0
    topic_dist = target_doc.get_topic_dist()

    with open(filepath, 'w', encoding = "utf-8") as f:
        f.write(f"--- Topic Distribution for Document #{doc_index} ---\n\n")

        # Optional: Write the original text so you can see if the math makes sense!
        # (Assuming your dataframe index matches the tomotopy document index)
        f.write(f"Original Text snippet: {df_w_texts['Full_Text'].iloc[doc_index][:200]}...\n\n")

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




def print_corpus_topic_distribution(mdl, topic_id_to_name, output_dir="output", country_name=""):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"corpus_topic_distribution_{country_name}.txt")

    # 1. Get the raw word counts for every topic
    topic_counts = mdl.get_count_by_topics()

    # 2. Calculate the total number of valid words the model processed
    total_words = sum(topic_counts)

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
            else:
                label = f"⬜ UNSEEDED (Topic {k_id})"
                
            # Append a tuple: (Percentage, Label, Raw Word Count, Topic ID)
            corpus_distribution.append((percentage, label, topic_counts[k_id], k_id))

        # 4. Sort the list from the largest topic to the smallest topic
        corpus_distribution.sort(reverse=True, key=lambda x: x[0])

        # 5. Write the formatted results WITH Top Words
        for percentage, label, count, k_id in corpus_distribution:
            # We only write topics that actually have words assigned to them
            if count > 0:
                # Extract the top 5 words for this specific topic
                top_words = [word for word, prob in mdl.get_topic_words(k_id, top_n=5)]
                
                # Write the stats and the keywords nicely formatted
                f.write(f"{label:<30} | {percentage:>5.2f}%  ({count:,} words)\n")
                f.write(f"   Keywords: {', '.join(top_words)}\n\n")