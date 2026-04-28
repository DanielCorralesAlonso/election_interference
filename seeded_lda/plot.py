import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os


def plot_topic_evolution(
    mdl,
    df_chunked,
    df_w_texts,
    topic_id_to_name,
    output_dir="output",
    country_name="",
    topics_to_plot=None,
):
    # ==========================================
    # 1. EXTRACT DATA & REASSEMBLE THE CHUNKS
    # ==========================================
    print("Extracting topic distributions...")

    os.makedirs(output_dir, exist_ok=True)

    # Create a list to hold the data for every chunk
    chunk_data = []

    # Ensure we have topics to work with for spike detection and plotting.
    if topics_to_plot is None:
        try:
            keys = sorted(topic_id_to_name.keys())
            topics_to_plot = keys[:3] if len(keys) >= 3 else keys
        except Exception:
            topics_to_plot = [0, 1, 2]

    if len(mdl.docs) != len(df_chunked):
        raise ValueError(
            f"mdl.docs ({len(mdl.docs)}) and df_chunked ({len(df_chunked)}) must have the same length. "
            "For country-specific plots, pass the full chunked dataframe and a filtered df_w_texts slice "
            "without resetting its index, or train a separate model for that country."
        )

    # Loop through every document in the tomotopy model
    for idx, doc in enumerate(mdl.docs):
        # Get the mathematical distribution across all topics for this specific chunk
        dist = doc.get_topic_dist()
        
        # Get the original article index from our chunked dataframe
        original_idx = df_chunked.iloc[idx]['Original_Index']
        
        # Store it as a dictionary
        row_data = {'Original_Index': original_idx}
        for k_id, prob in enumerate(dist):
            row_data[f'Topic_{k_id}'] = prob
            
        chunk_data.append(row_data)

    # Convert to a DataFrame
    df_dists = pd.DataFrame(chunk_data)

    # Because one article was split into multiple chunks, we group by the Original_Index
    # and take the mean to reconstruct the article's total topic distribution.
    df_article_topics = df_dists.groupby('Original_Index').mean().reset_index()

    # ==========================================
    # 2. MERGE WITH THE ORIGINAL TIMESTAMPS
    # ==========================================
    # Merge the topic distributions back into your original dataframe using the index
    # (Assuming your original dataframe has a column named 'Event_Date' that contains the publication date)
    df_final = df_w_texts[['Event_Date']].merge(
        df_article_topics, 
        left_index=True, 
        right_on='Original_Index'
    )

    # Ensure the date column is officially recognized by pandas as a Datetime object
    df_final['Event_Date'] = pd.to_datetime(
        df_final['Event_Date'].astype(str), 
        format='%Y%m%d'
    )

    # ==========================================
    # 3. AGGREGATE BY WEEK & SMOOTH
    # ==========================================
    # Set the date as the index so we can do time-series math
    df_final.set_index('Event_Date', inplace=True)

    topic_cols = [col for col in df_final.columns if col.startswith('Topic_')]

    # Group the data by Week ('W') and calculate the mean topic probability for that week
    weekly_trends = df_final[topic_cols].resample('W').mean()

    # Apply a 1-week rolling average to smooth out the jagged spikes
    # min_periods=1 ensures the graph doesn't drop the first two weeks
    smoothed_trends = weekly_trends.rolling(window=1, min_periods=1).mean()

    # ==========================================
    # 3.5 SPIKE DETECTION & EXPORT
    # ==========================================
    # For each selected topic, find the top 3 weekly peaks and save the
    # original articles occurring in the week of the peak whose topic
    # probability for any topic is > 0.05.
    spikes_dir = os.path.join(output_dir, "spikes")
    os.makedirs(spikes_dir, exist_ok=True)
    spikes_path = os.path.join(spikes_dir, f"spikes_{country_name.replace(' ','_')}.txt")

    # Work with a copy that still has article-level rows and Event_Date as column
    df_articles = df_final.reset_index()

    THRESHOLD = 0.05
    TOP_K = 3

    with open(spikes_path, 'w', encoding='utf-8') as fh:
        fh.write(f"Spikes report for {country_name}\n")
        fh.write("=" * 60 + "\n\n")

        for k_id in topics_to_plot:
            col = f"Topic_{k_id}"
            if col not in smoothed_trends.columns:
                continue

            fh.write(f"Topic {k_id} - {topic_id_to_name.get(k_id, str(k_id))}\n")
            fh.write('-' * 60 + "\n")

            # get the top TOP_K weekly spike dates and their magnitudes
            top_spikes = smoothed_trends[col].nlargest(TOP_K)
            if top_spikes.empty:
                fh.write("No spikes found.\n\n")
                continue

            for spike_dt, spike_val in top_spikes.items():
                # define week window: last 7 days up to spike_dt
                week_start = spike_dt - pd.Timedelta(days=6)
                week_end = spike_dt

                fh.write(f"Spike week ending {spike_dt.date()} (value={spike_val:.4f})\n")

                # articles in that week
                week_articles = df_articles[(df_articles['Event_Date'] >= week_start) & (df_articles['Event_Date'] <= week_end)]
                if week_articles.empty:
                    fh.write("  No articles in this week.\n\n")
                    continue

                # find articles in that week where this topic has non-trivial mass
                matched = week_articles[week_articles[col] > THRESHOLD].copy()
                if matched.empty:
                    fh.write("  No articles with topic prob > {THRESHOLD} in this week.\n\n")
                    continue

                # For each matched article, write metadata, topic distribution (>THRESHOLD), and the raw text
                for _, row in matched.iterrows():
                    orig_idx = int(row['Original_Index'])
                    event_date = row['Event_Date']

                    # collect topic probs > THRESHOLD
                    topic_probs = {t: float(row[t]) for t in topic_cols if (t in row.index and row[t] > THRESHOLD)}

                    # try to get the original text from df_w_texts (should preserve original indices)
                    full_text = None
                    try:
                        full_text = str(df_w_texts.loc[orig_idx, 'Full_Text'])
                    except Exception:
                        # fallback: try matching by Original_Index in df_chunked
                        try:
                            candidate_rows = df_chunked[df_chunked['Original_Index'] == orig_idx]
                            if not candidate_rows.empty and 'Full_Text' in candidate_rows.columns:
                                full_text = str(candidate_rows.iloc[0]['Full_Text'])
                        except Exception:
                            full_text = ''

                    fh.write(f"  Article index: {orig_idx} | Date: {event_date.date()}\n")
                    fh.write(f"  Topic probs (> {THRESHOLD}): {topic_probs}\n")
                    fh.write("  ---\n")
                    fh.write(full_text.replace('\n', ' ')[:10000] + "\n")
                    fh.write("  " + ('-' * 40) + "\n\n")

            fh.write("\n")

    # ==========================================
    # 4. PLOTTING THE EVOLUTION
    # ==========================================
    # Set up a large, clean figure
    plt.figure(figsize=(14, 7))

    # Choose which specific topics you want to plot so it isn't too cluttered
    # Let's assume you want to plot your seeded topics (e.g., IDs 0, 1, and 2)
    # You can look up your exact IDs from your `topic_id_to_name` mapping
    if topics_to_plot is None:
        topics_to_plot = [0, 1, 2]

    # Define some nice colors
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    for i, k_id in enumerate(topics_to_plot):
        # Get the human-readable label
        label = topic_id_to_name.get(k_id, f"Topic {k_id}")
        
        # Plot the smoothed line
        # Multiply by 100 to convert decimals to percentages (e.g., 0.25 -> 25%)
        plt.plot(
            smoothed_trends.index, 
            smoothed_trends[f'Topic_{k_id}'] * 100, 
            label=label, 
            linewidth=3,
            color=colors[i % len(colors)]
        )

    # Format the Graph visually
    plt.title(f"Evolution of political narratives in the news - {country_name}", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Percentage of Weekly News Coverage (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)

    # Format the X-axis dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Show a tick every 1 month
    plt.xticks(rotation=45)

    # Add a grid, legend, and layout
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Seeded Topics", fontsize=11, title_fontsize=12, loc='upper left')
    plt.tight_layout()

    # Save or display the plot
    plt.savefig(os.path.join(output_dir, f"topic_evolution_{country_name}.png"), dpi=300)
    plt.close()