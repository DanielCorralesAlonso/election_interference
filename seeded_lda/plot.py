import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

# Make all plots publication-ready with larger fonts
plt.rcParams.update({
    'font.size': 14,          # Global font size
    'axes.titlesize': 16,     # Subplot titles
    'axes.labelsize': 14,     # X/Y axis labels
    'xtick.labelsize': 12,    # X tick marks
    'ytick.labelsize': 12,    # Y tick marks
    'legend.fontsize': 12,    # Legend text
    'figure.titlesize': 18    # Main overarching title
})



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

    def topic_label(k_id):
        if k_id in topic_id_to_name:
            return topic_id_to_name[k_id]
        try:
            top_words = mdl.get_topic_words(k_id, top_n=3)
            words = [w for w, _ in top_words]
            if words:
                return " / ".join(words)
        except Exception:
            pass
        return f"Topic_{k_id}"

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

    # Group by day and compute mean topic probability for each day.
    # Fill missing days with zeros so plots remain continuous when no articles were published.
    daily_trends = df_final[topic_cols].resample('D').mean().fillna(0)

    # Smooth with a 7-day rolling window so spikes are still day-resolved.
    smoothed_trends = daily_trends.rolling(window=7, min_periods=1).mean()

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

            fh.write(f"Topic {k_id} - {topic_label(k_id)}\n")
            fh.write('-' * 60 + "\n")

            # Get the top TOP_K day-level spike dates and their magnitudes.
            top_spikes = smoothed_trends[col].nlargest(TOP_K)
            if top_spikes.empty:
                fh.write("No spikes found.\n\n")
                continue

            for spike_dt, spike_val in top_spikes.items():
                fh.write(f"Spike day {spike_dt.date()} (7-day smoothed value={spike_val * 100:.1f}%)\n")

                # Articles for the exact spike day.
                day_articles = df_articles[df_articles['Event_Date'].dt.date == spike_dt.date()]
                if day_articles.empty:
                    fh.write("  No articles on this day.\n\n")
                    continue

                # Find articles for this day where this topic has non-trivial mass.
                matched = day_articles[day_articles[col] > THRESHOLD].copy()
                if matched.empty:
                    fh.write(f"  No articles with topic prob > {THRESHOLD:.2f} on this day.\n\n")
                    continue

                # For each matched article, write metadata, topic distribution (>THRESHOLD), and the raw text
                for _, row in matched.iterrows():
                    orig_idx = int(row['Original_Index'])
                    event_date = row['Event_Date']

                    # Collect topic probabilities above threshold with seeded topic names.
                    topic_probs = {
                        topic_label(int(t.split('_')[1])): f"{row[t] * 100:.1f}%"
                        for t in topic_cols
                        if (t in row.index and row[t] > THRESHOLD)
                    }

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





def plot_document_length_distribution(df_w_texts, text_col, output_dir="output", country_name="", title="Document Length Distribution"):
    plt.figure(figsize=(10, 6))
    df_w_texts['Doc_Length'] = df_w_texts[text_col].apply(len)
    plt.hist(df_w_texts['Doc_Length'], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{title} - {country_name}", fontsize=16)
    plt.xlabel("Number of Tokens", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"doc_length_distribution_{country_name}.png"), dpi=300)

def plot_topic_evolution_comparison(
    mdl,
    df_chunked,
    df_w_texts,
    topic_id_to_name,
    output_dir="output",
    countries=("Russia", "China", "Iran"),
    topics_to_plot=None,
    filename="topic_evolution_russia_china_iran.png",
):
    """Plot topic evolution for multiple countries in stacked panels with a shared y-axis."""
    os.makedirs(output_dir, exist_ok=True)

    country_aliases = {
        "russia": {"russia", "ru", "rus", "russian federation", "russian fed."},
        "china": {"china", "cn", "chn", "people's republic of china", "prc", "p.r.c."},
        "iran": {"iran", "ir", "irn", "iran (islamic republic of)", "islamic republic of iran"},
    }

    def normalize_value(value):
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    def country_matches(series, target_country):
        aliases = set()
        canonical = normalize_value(target_country)
        aliases.add(canonical)
        aliases.update(country_aliases.get(canonical, set()))
        normalized = series.astype(str).map(normalize_value)
        return normalized.isin(aliases)

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

    def build_smoothed_trends(country_df):
        chunk_data = []
        for idx, doc in enumerate(mdl.docs):
            dist = doc.get_topic_dist()
            original_idx = df_chunked.iloc[idx]["Original_Index"]
            row_data = {"Original_Index": original_idx}
            for k_id, prob in enumerate(dist):
                row_data[f"Topic_{k_id}"] = prob
            chunk_data.append(row_data)

        df_dists = pd.DataFrame(chunk_data)
        df_article_topics = df_dists.groupby("Original_Index").mean().reset_index()

        event_col = "Event_Date" if "Event_Date" in country_df.columns else None
        if event_col is None:
            return None

        df_final = country_df[[event_col]].merge(
            df_article_topics,
            left_index=True,
            right_on="Original_Index",
        )
        df_final["Event_Date"] = pd.to_datetime(df_final["Event_Date"].astype(str), errors="coerce")
        if df_final["Event_Date"].isna().all():
            df_final["Event_Date"] = pd.to_datetime(df_final["Event_Date"].astype(str).str.replace(r"[^0-9]", "", regex=True), format="%Y%m%d", errors="coerce")
        df_final = df_final.dropna(subset=["Event_Date"])
        if df_final.empty:
            return None
        df_final.set_index("Event_Date", inplace=True)

        topic_cols = [col for col in df_final.columns if col.startswith("Topic_")]
        daily_trends = df_final[topic_cols].resample("D").mean().fillna(0)
        return daily_trends.rolling(window=7, min_periods=1).mean()

    country_frames = []
    if "Country" not in df_w_texts.columns:
        print("Warning: no Country column found. Skipping comparison plot.")
        return

    for country in countries:
        mask = country_matches(df_w_texts["Country"], country)
        country_df = df_w_texts[mask]
        if country_df.empty:
            continue
        trends = build_smoothed_trends(country_df)
        if trends is None or trends.empty:
            continue
        country_frames.append((country, trends))

    if not country_frames:
        print("Warning: no matching country rows found for the comparison plot. Skipping.")
        return

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    all_values = []
    for _, trends in country_frames:
        for k_id in topics_to_plot:
            col = f"Topic_{k_id}"
            if col in trends.columns:
                all_values.append(trends[col] * 100)

    if all_values:
        combined = pd.concat(all_values, axis=0)
        y_min = float(combined.min())
        y_max = float(combined.max())
    else:
        y_min, y_max = 0.0, 1.0

    fig, axes = plt.subplots(len(country_frames), 1, figsize=(14, 4.5 * len(country_frames)), sharex=True, sharey=True)
    if len(country_frames) == 1:
        axes = [axes]

    for ax, (country, trends) in zip(axes, country_frames):
        for i, k_id in enumerate(topics_to_plot):
            col = f"Topic_{k_id}"
            if col not in trends.columns:
                continue
            label = topic_id_to_name.get(k_id, f"Topic {k_id}")
            ax.plot(
                trends.index,
                trends[col] * 100,
                label=label,
                linewidth=2.5,
                color=colors[i % len(colors)],
            )
        ax.set_title(country, fontsize=13, fontweight="bold")
        ax.set_ylabel("Topic share (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(y_min, y_max)

    axes[-1].set_xlabel("Date", fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Seeded Topics", loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("Topic evolution comparison: Russia, China, and Iran", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison topic evolution plot to {output_path}")
    plt.close(fig)
    plt.close()