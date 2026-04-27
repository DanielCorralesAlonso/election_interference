import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os


def plot_topic_evolution(mdl, df_chunked, df_w_texts, topic_id_to_name, output_dir="output", country_name=""):
    # ==========================================
    # 1. EXTRACT DATA & REASSEMBLE THE CHUNKS
    # ==========================================
    print("Extracting topic distributions...")

    # Create a list to hold the data for every chunk
    chunk_data = []

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

    # Group the data by Week ('W') and calculate the mean topic probability for that week
    weekly_trends = df_final.resample('W').mean()

    # Apply a 3-week rolling average to smooth out the jagged spikes
    # min_periods=1 ensures the graph doesn't drop the first two weeks
    smoothed_trends = weekly_trends.rolling(window=1, min_periods=1).mean()

    # ==========================================
    # 4. PLOTTING THE EVOLUTION
    # ==========================================
    # Set up a large, clean figure
    plt.figure(figsize=(14, 7))

    # Choose which specific topics you want to plot so it isn't too cluttered
    # Let's assume you want to plot your seeded topics (e.g., IDs 0, 1, and 2)
    # You can look up your exact IDs from your `topic_id_to_name` mapping
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