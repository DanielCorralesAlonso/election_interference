import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
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



def _posterior_weekly_std(theta_samples, df_w_texts, weekly_index):
    """
    Compute posterior std of the weekly mean topic probability from thinned
    MCMC samples.

    theta_samples : np.ndarray (n_samples, n_docs, K)
    df_w_texts    : DataFrame with 'Event_Date' column and index aligned to n_docs
    weekly_index  : DatetimeIndex of weekly bins (from the main resample call)

    Returns a DataFrame (n_weeks, K) of posterior std values, aligned to weekly_index.
    """
    n_samples, n_docs, K = theta_samples.shape
    valid_indices = list(df_w_texts.index)
    topic_cols = [f'Topic_{k}' for k in range(K)]

    sample_means = []
    for s in range(n_samples):
        rows = [
            {'Original_Index': idx,
             **{f'Topic_{k}': theta_samples[s, idx, k] for k in range(K)}}
            for idx in valid_indices if idx < n_docs
        ]
        df_s = pd.DataFrame(rows)
        df_m = df_w_texts[['Event_Date']].merge(df_s, left_index=True, right_on='Original_Index')
        df_m['Event_Date'] = pd.to_datetime(df_m['Event_Date'].astype(str), format='%Y%m%d', errors='coerce')
        df_m = df_m.dropna(subset=['Event_Date']).set_index('Event_Date')
        wm = df_m[topic_cols].resample('W').mean().reindex(weekly_index).fillna(0)
        sample_means.append(wm.values)

    stacked = np.stack(sample_means, axis=0)          # (n_samples, n_weeks, K)
    std_vals = stacked.std(axis=0)                     # (n_weeks, K)
    return pd.DataFrame(std_vals, index=weekly_index, columns=topic_cols)


def _draw_election_lines(ax, election_dates):
    """Draw a dashed vertical line + rotated label for each election date.

    election_dates : dict[str, str]  e.g. {"US Election": "2024-11-05"}
    Uses x-data / y-axes coordinates so labels always sit at the top of the
    panel regardless of y-scale.
    """
    if not election_dates:
        return
    trans = ax.get_xaxis_transform()  # x: data coords, y: axes [0, 1]
    for label, date_str in election_dates.items():
        dt = pd.to_datetime(date_str)
        ax.axvline(dt, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.text(dt, 0.99, label, rotation=90, fontsize=7,
                va='top', ha='right', alpha=0.75, transform=trans)


def plot_topic_evolution(
    mdl,
    df_w_texts,
    topic_id_to_name,
    output_dir="output",
    country_name="",
    topics_to_plot=None,
    top_n_prominence=5,
    theta_samples=None,
    show_prominence=False,
    show_uncertainty=False,
    election_dates=None,
):
    print("Extracting topic distributions...")

    os.makedirs(output_dir, exist_ok=True)

    chunk_data = []

    if topics_to_plot is None:
        try:
            topics_to_plot = sorted(topic_id_to_name.keys())
        except Exception:
            topics_to_plot = [0, 1, 2]

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

    # Build topic distributions filtered to articles present in df_w_texts.
    # mdl.docs may be a MultiChainSummary (averaged) or a raw tomotopy model.
    # Iterate via enumerate rather than integer subscripting: tomotopy's DocList
    # len() includes documents whose tokens were all filtered by min_cf, but
    # direct indexing into those positions raises IndexError.
    valid_indices = set(df_w_texts.index)
    for idx, doc in enumerate(mdl.docs):
        if idx not in valid_indices:
            continue
        try:
            dist = doc.get_topic_dist()
        except (IndexError, Exception):
            continue
        row_data = {'Original_Index': idx}
        for k_id, prob in enumerate(dist):
            row_data[f'Topic_{k_id}'] = prob
        chunk_data.append(row_data)

    df_article_topics = pd.DataFrame(chunk_data)

    # Build prominence matrix: 1 if topic is in top-N for that article, else 0
    _all_topic_cols = [c for c in df_article_topics.columns if c.startswith('Topic_')]
    _prob_vals = df_article_topics[_all_topic_cols].values
    _top_idx = np.argsort(_prob_vals, axis=1)[:, -top_n_prominence:]
    _prom_vals = np.zeros_like(_prob_vals)
    for _i, _ti in enumerate(_top_idx):
        _prom_vals[_i, _ti] = 1.0
    df_article_prominence = pd.DataFrame(_prom_vals, columns=_all_topic_cols)
    df_article_prominence['Original_Index'] = df_article_topics['Original_Index'].values

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

    weekly_mean     = df_final[topic_cols].resample('W').mean()
    smoothed_trends = weekly_mean.fillna(0)

    if show_uncertainty:
        if theta_samples is not None:
            weekly_uncertainty = _posterior_weekly_std(theta_samples, df_w_texts, smoothed_trends.index)
        else:
            weekly_std         = df_final[topic_cols].resample('W').std()
            weekly_count       = df_final[topic_cols].resample('W').count()
            weekly_uncertainty = weekly_std / weekly_count.pow(0.5)
    else:
        weekly_uncertainty = None

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
                fh.write(f"Spike week ending {spike_dt.date()} (weekly value={spike_val * 100:.1f}%)\n")

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
                    try:
                        full_text = str(df_w_texts.loc[orig_idx, 'Full_Text'])
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

    colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    for i, k_id in enumerate(topics_to_plot):
        label = topic_id_to_name.get(k_id, f"Topic {k_id}")
        col   = f'Topic_{k_id}'
        mean  = smoothed_trends[col] * 100
        color = colors[i % len(colors)]

        plt.plot(smoothed_trends.index, mean, label=label, linewidth=3, color=color)
        if weekly_uncertainty is not None and col in weekly_uncertainty.columns:
            unc = weekly_uncertainty[col].fillna(0) * 100
            plt.fill_between(smoothed_trends.index, mean - 2*unc, mean + 2*unc, alpha=0.15, color=color)

    # Format the Graph visually
    plt.title(f"Evolution of political narratives in the news - {country_name}", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Weekly topic share (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)

    # Format the X-axis dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Show a tick every 1 month
    plt.xticks(rotation=45)

    # Add a grid, legend, and layout
    plt.grid(True, linestyle='--', alpha=0.6)
    _draw_election_lines(plt.gca(), election_dates)
    plt.legend(
        title="Topics",
        fontsize=10,
        title_fontsize=11,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(topics_to_plot),
        frameon=True,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    # Save or display the plot
    plt.savefig(os.path.join(output_dir, f"topic_evolution_{country_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 5. PROMINENCE PLOT (top-N binary) — optional
    # ==========================================
    if not show_prominence:
        return

    df_prom_final = df_w_texts[['Event_Date']].merge(
        df_article_prominence,
        left_index=True,
        right_on='Original_Index',
    )
    df_prom_final['Event_Date'] = pd.to_datetime(
        df_prom_final['Event_Date'].astype(str), format='%Y%m%d'
    )
    df_prom_final.set_index('Event_Date', inplace=True)

    prom_topic_cols = [c for c in df_prom_final.columns if c.startswith('Topic_')]
    smoothed_prom = df_prom_final[prom_topic_cols].resample('W').mean().fillna(0)

    plt.figure(figsize=(14, 7))
    for i, k_id in enumerate(topics_to_plot):
        col = f'Topic_{k_id}'
        if col not in smoothed_prom.columns:
            continue
        label = topic_id_to_name.get(k_id, f'Topic {k_id}')
        plt.plot(
            smoothed_prom.index,
            smoothed_prom[col] * 100,
            label=label,
            linewidth=3,
            color=colors[i % len(colors)],
        )

    plt.title(f"Topic prominence in news coverage — {country_name}", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel(f"Articles with topic in top-{top_n_prominence} (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(
        title="Topics",
        fontsize=10,
        title_fontsize=11,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(topics_to_plot),
        frameon=True,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(os.path.join(output_dir, f"topic_prominence_{country_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()



def plot_document_entropy_by_country(mdl, df_w_texts, output_dir="output", country_name="all"):
    """
    Compute the normalised Shannon entropy of each document's topic distribution
    and compare distributions across countries via violin + strip plots.

    Normalised entropy = H(theta_d) / log(K), where H = -sum(p * log(p)).
    A value of 1.0 = perfectly uniform across all K topics (maximally diffuse).
    A value near 0  = near-deterministic (one topic dominates).

    Saves both a plot and a CSV of per-country summary statistics.
    """
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    K = mdl.k
    log_K = np.log(K)

    valid_indices = set(df_w_texts.index)
    rows = []
    for idx, doc in enumerate(mdl.docs):
        if idx not in valid_indices:
            continue
        try:
            theta = np.array(doc.get_topic_dist())
        except Exception:
            continue
        theta = np.clip(theta, 1e-12, 1.0)
        entropy = -float(np.sum(theta * np.log(theta)))
        norm_entropy = entropy / log_K
        rows.append({"Original_Index": idx, "Entropy": entropy, "Norm_Entropy": norm_entropy})

    df_ent = pd.DataFrame(rows)

    if "Country" in df_w_texts.columns:
        df_ent = df_ent.merge(
            df_w_texts[["Country"]].reset_index().rename(columns={"index": "Original_Index"}),
            on="Original_Index",
            how="left",
        )
        group_col = "Country"
    else:
        df_ent["Country"] = country_name
        group_col = "Country"

    # Summary statistics
    summary = df_ent.groupby(group_col)["Norm_Entropy"].describe(percentiles=[0.25, 0.5, 0.75])
    summary_path = os.path.join(output_dir, f"entropy_by_country_{country_name}.csv")
    summary.to_csv(summary_path)
    print(f"\nDocument entropy summary (normalised, K={K}):")
    print(summary.to_string())

    # Plot
    countries_present = df_ent[group_col].dropna().unique()
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(countries_present)), 6))

    sns.violinplot(
        data=df_ent, x=group_col, y="Norm_Entropy",
        inner=None, palette="Set2", alpha=0.6, ax=ax, order=sorted(countries_present),
    )
    sns.stripplot(
        data=df_ent, x=group_col, y="Norm_Entropy",
        color="black", alpha=0.15, size=2.5, jitter=True, ax=ax, order=sorted(countries_present),
    )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label=f"Max entropy (uniform over {K} topics)")
    ax.set_xlabel("Country", fontsize=13)
    ax.set_ylabel(f"Normalised entropy  H(θ) / log({K})", fontsize=13)
    ax.set_title("Per-document topic entropy by country\n(higher = more diffuse topic mix per article)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"entropy_by_country_{country_name}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Entropy plot saved to {plot_path}")

    return df_ent


def plot_topic_cooccurrence(mdl, output_dir="output", topic_id_to_name=None,
                            country_name="all", top_pairs=20):
    """
    Heatmap (+ ranked summary) of topic-topic Pearson correlation across
    CLR-transformed document-topic mixtures (theta).

    Every other plot in this module is "marginal": it tracks one topic's share
    in isolation, over time or across documents/countries. None of them show
    *structure* — whether topics travel together within the same articles. This
    fills that gap: two topics get a high positive correlation here when
    articles that lean heavily on one also lean on the other (a shared framing
    or "narrative bundle"), and a negative correlation when they compete for
    the same documents (mutually exclusive framings). For a seeded model this
    is also a direct check on seed separation — e.g. do the "russia" and
    "hacking" seeded topics co-occur more than chance, or has the seeding
    failed to keep them distinct?

    WHY CLR INSTEAD OF RAW PEARSON ON THETA
    ----------------------------------------
    LDA topic vectors theta_d are compositional: they sum to 1 for every
    document. This simplex constraint creates a spurious suppression of
    Pearson correlations. In a K-topic model, the average off-diagonal
    Pearson r is bounded below by -1/(K-1): with K=30 topics that floor is
    about -0.033, meaning correlations are mathematically squeezed toward
    zero regardless of the true co-occurrence structure. The result is a
    globally faint heatmap where even strongly co-occurring topics appear
    near-zero.

    The principled fix is the centred log-ratio (CLR) transform (Aitchison,
    1982). For each document d:

        clr(theta_d)_k = log(theta_d_k) - (1/K) * sum_j log(theta_d_j)

    CLR lifts the vectors off the simplex into unconstrained Euclidean space
    while preserving relative differences between topic weights. Pearson
    correlation on CLR coordinates is then interpretable: a positive r means
    documents that are above-average on topic i (relative to their own
    mixture) also tend to be above-average on topic j — genuine co-occurrence.
    A negative r means the two topics genuinely compete for the same
    documents. Neither direction is artificially compressed by the sum-to-one
    constraint.

    Reference: Aitchison, J. (1982). The statistical analysis of
    compositional data. Journal of the Royal Statistical Society: Series B,
    44(2), 139-160.

    Saves a full K x K heatmap plus a plain-text ranking of the strongest
    pairs.
    """
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    topic_id_to_name = topic_id_to_name or {}

    theta = np.array([doc.get_topic_dist() for doc in mdl.docs])
    K = theta.shape[1]

    # CLR transform: log(theta_k) minus the per-document geometric mean of all
    # log(theta). The small epsilon prevents log(0) for near-zero probability slots.
    # This removes the simplex constraint (sum-to-1) that suppresses raw Pearson r.
    log_theta = np.log(theta + 1e-10)
    clr_theta = log_theta - log_theta.mean(axis=1, keepdims=True)
    corr = np.corrcoef(clr_theta, rowvar=False)

    # Self-correlation is trivially 1.0 and would otherwise dominate the
    # colour scale as a solid diagonal — mask it so the off-diagonal structure
    # (the actually informative part) stands out.
    corr_plot = corr.copy()
    np.fill_diagonal(corr_plot, np.nan)

    def _label(k):
        if k in topic_id_to_name:
            return f"[{topic_id_to_name[k]}]"
        return mdl.get_topic_words(k, top_n=1)[0][0]

    labels = [_label(k) for k in range(K)]

    fig, ax = plt.subplots(figsize=(max(10, K * 0.22), max(8, K * 0.22)))
    sns.heatmap(
        corr_plot, xticklabels=labels, yticklabels=labels, cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True, mask=np.isnan(corr_plot),
        cbar_kws={"label": "Pearson r (CLR-transformed θ)"}, ax=ax,
    )
    ax.set_title(
        "Which narratives travel together within the same articles?\n"
        f"Pearson r of CLR(θ) — {country_name}   (seeded topics in brackets)\n"
        "CLR removes simplex bias; r > 0 = co-occurring framings, r < 0 = competing framings",
        fontsize=11,
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"topic_cooccurrence_{country_name}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Named ranking of the strongest pairs — turns "blocks in the heatmap" into
    # "topic A pairs with topic B", readable without squinting at tick labels.
    pairs = [(corr[i, j], i, j) for i in range(K) for j in range(i + 1, K)]
    pairs.sort(key=lambda t: -abs(t[0]))

    summary_path = os.path.join(output_dir, f"topic_cooccurrence_{country_name}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Strongest topic co-occurrences by |Pearson r| of theta — {country_name} (K={K})\n\n")
        for r, i, j in pairs[:top_pairs]:
            tag_i = f"SEEDED[{topic_id_to_name[i]}]" if i in topic_id_to_name else f"#{i} ({labels[i]})"
            tag_j = f"SEEDED[{topic_id_to_name[j]}]" if j in topic_id_to_name else f"#{j} ({labels[j]})"
            relation = "co-occur (shared framing)" if r > 0 else "mutually exclusive (competing framings)"
            f.write(f"  r={r:+.3f}  {tag_i}  <->  {tag_j}   [{relation}]\n")

    print(f"Topic co-occurrence heatmap saved to {plot_path}")
    print(f"Topic co-occurrence summary saved to {summary_path}")
    return corr


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
    df_w_texts,
    topic_id_to_name,
    output_dir="output",
    countries=("Russia", "China", "Iran"),
    topics_to_plot=None,
    filename="topic_evolution_russia_china_iran.png",
    theta_samples=None,
    show_uncertainty=False,
    election_dates=None,
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
            topics_to_plot = sorted(topic_id_to_name.keys())
        except Exception:
            topics_to_plot = [0, 1, 2]

    if len(mdl.docs) != len(df_w_texts):
        raise ValueError(
            f"mdl.docs ({len(mdl.docs)}) and df_w_texts ({len(df_w_texts)}) must have the same length."
        )

    # Build full topic distribution dataframe once, then filter per country
    all_dists = []
    for idx, doc in enumerate(mdl.docs):
        dist = doc.get_topic_dist()
        row_data = {"Original_Index": idx}
        for k_id, prob in enumerate(dist):
            row_data[f"Topic_{k_id}"] = prob
        all_dists.append(row_data)
    df_all_topics = pd.DataFrame(all_dists)

    def build_smoothed_trends(country_df):
        event_col = "Event_Date" if "Event_Date" in country_df.columns else None
        if event_col is None:
            return None

        country_indices = set(country_df.index)
        df_country_topics = df_all_topics[df_all_topics["Original_Index"].isin(country_indices)]

        df_final = country_df[[event_col]].merge(
            df_country_topics,
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

        topic_cols  = [col for col in df_final.columns if col.startswith("Topic_")]
        weekly_mean = df_final[topic_cols].resample("W").mean().fillna(0)

        if show_uncertainty:
            if theta_samples is not None:
                unc = _posterior_weekly_std(theta_samples, country_df, weekly_mean.index)
            else:
                weekly_std   = df_final[topic_cols].resample("W").std()
                weekly_count = df_final[topic_cols].resample("W").count()
                unc = (weekly_std / weekly_count.pow(0.5)).fillna(0)
        else:
            unc = None

        return weekly_mean, unc

    country_frames = []
    if "Country" not in df_w_texts.columns:
        print("Warning: no Country column found. Skipping comparison plot.")
        return

    for country in countries:
        mask = country_matches(df_w_texts["Country"], country)
        country_df = df_w_texts[mask]
        if country_df.empty:
            continue
        result = build_smoothed_trends(country_df)
        if result is None:
            continue
        trends, unc = result
        if trends.empty:
            continue
        country_frames.append((country, trends, unc))

    if not country_frames:
        print("Warning: no matching country rows found for the comparison plot. Skipping.")
        return

    colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    all_values = []
    for _, trends, _ in country_frames:
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

    for ax, (country, trends, unc) in zip(axes, country_frames):
        for i, k_id in enumerate(topics_to_plot):
            col = f"Topic_{k_id}"
            if col not in trends.columns:
                continue
            label     = topic_id_to_name.get(k_id, f"Topic {k_id}")
            color     = colors[i % len(colors)]
            mean_vals = trends[col] * 100
            ax.plot(trends.index, mean_vals, label=label, linewidth=2.5, color=color)
            if unc is not None and col in unc.columns:
                unc_vals = unc[col].fillna(0) * 100
                ax.fill_between(trends.index, mean_vals - 2*unc_vals, mean_vals + 2*unc_vals, alpha=0.15, color=color)
        ax.set_title(country, fontsize=13, fontweight="bold")
        ax.set_ylabel("Topic share (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(y_min, y_max)
        _draw_election_lines(ax, election_dates)

    axes[-1].set_xlabel("Date", fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            title="Topics",
            fontsize=10,
            title_fontsize=11,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.35),
            bbox_transform=axes[-1].transAxes,
            ncol=len(handles),
            frameon=True,
        )

    fig.suptitle("Topic evolution comparison: Russia, China, and Iran", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(bottom=0.18)
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison topic evolution plot to {output_path}")
    plt.close(fig)
    plt.close()



_ACTOR_COLORS = {
    'russia': '#d62728',  # red
    'china':  '#1f77b4',  # blue
    'iran':   '#2ca02c',  # green
}
_FALLBACK_COLORS = ['#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_narrative_by_country(
    mdl,
    df_w_texts,
    topic_id_to_name,
    narrative_topic_ids,
    output_dir="output",
    country_name="all",
    n_cols=4,
    show_uncertainty=False,
    election_dates=None,
):
    """
    For each narrative topic, plots the weekly mean θ for every country in the
    corpus on the same axes, enabling direct cross-country comparison of how
    the same narrative evolves differently in each country's coverage.

    Grid layout: one panel per narrative topic, one line per country.
    Russia=red, China=blue, Iran=green; other countries get fallback colours.
    Uses the Country column for grouping (metadata, not model-derived).
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'Country' not in df_w_texts.columns:
        print('plot_narrative_by_country: no Country column, skipping.')
        return

    countries = sorted(df_w_texts['Country'].dropna().unique())
    if not countries:
        return

    fallback_iter = iter(_FALLBACK_COLORS)
    color_map = {}
    for c in countries:
        key = str(c).strip().lower()
        color_map[c] = _ACTOR_COLORS.get(key, next(fallback_iter, '#333333'))

    valid_indices = set(df_w_texts.index)
    rows = []
    for idx, doc in enumerate(mdl.docs):
        if idx not in valid_indices:
            continue
        try:
            dist = doc.get_topic_dist()
        except Exception:
            continue
        row = {'Original_Index': idx}
        for k_id, prob in enumerate(dist):
            row[f'Topic_{k_id}'] = prob
        rows.append(row)
    df_theta = pd.DataFrame(rows)

    df_merged = df_w_texts[['Event_Date', 'Country']].merge(
        df_theta, left_index=True, right_on='Original_Index'
    )
    df_merged['Event_Date'] = pd.to_datetime(
        df_merged['Event_Date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    df_merged = df_merged.dropna(subset=['Event_Date']).set_index('Event_Date')

    n_topics = len(narrative_topic_ids)
    n_cols   = min(n_cols, n_topics)
    n_rows   = int(np.ceil(n_topics / n_cols))

    panel_w, panel_h = 4.5, 3.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten() if n_topics > 1 else [axes]

    for ax_idx, nid in enumerate(narrative_topic_ids):
        ax  = axes_flat[ax_idx]
        col = f'Topic_{nid}'
        narrative_label = topic_id_to_name.get(nid, f'Topic {nid}')

        if col not in df_merged.columns:
            ax.set_visible(False)
            continue

        for country in countries:
            country_data = df_merged[df_merged['Country'] == country]
            if country_data.empty:
                continue
            weekly_mean = country_data[col].resample('W').mean().fillna(0)
            mean_vals   = weekly_mean * 100
            color       = color_map[country]
            ax.plot(weekly_mean.index, mean_vals, label=country, linewidth=1.8, color=color)
            if show_uncertainty:
                weekly_std   = country_data[col].resample('W').std()
                weekly_count = country_data[col].resample('W').count()
                se_vals      = (weekly_std / weekly_count.pow(0.5)).fillna(0) * 100
                ax.fill_between(
                    weekly_mean.index, mean_vals - 2 * se_vals, mean_vals + 2 * se_vals,
                    alpha=0.12, color=color,
                )

        ax.set_title(narrative_label, fontsize=10, fontweight='bold')
        ax.set_ylabel('Topic share (%)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        _draw_election_lines(ax, election_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)

    for ax_idx in range(n_topics, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            title='Country',
            fontsize=10,
            title_fontsize=11,
            loc='outside lower center',
            ncol=len(countries),
            frameon=True,
        )

    fig.suptitle(
        f'Cross-country narrative comparison — {country_name}',
        fontsize=14, fontweight='bold',
    )
    plot_path = os.path.join(output_dir, f'narrative_by_country_{country_name}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Cross-country narrative comparison plot saved to {plot_path}')


def plot_narrative_stacked_area(
    mdl,
    df_w_texts,
    topic_id_to_name,
    narrative_topic_ids,
    output_dir="output",
    country_name="all",
    election_dates=None,
):
    """
    Stacked area chart of narrative topic composition over time.

    Each week the selected topics are normalised to sum to 100%, showing the
    relative share of narrative attention each topic commands — making it easy
    to see when one narrative crowds out another. Topics are ordered from most
    to least prominent (by average weekly share) so the dominant topics form a
    stable base layer.
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_indices = set(df_w_texts.index)
    rows = []
    for idx, doc in enumerate(mdl.docs):
        if idx not in valid_indices:
            continue
        try:
            dist = doc.get_topic_dist()
        except Exception:
            continue
        row = {'Original_Index': idx}
        for k_id, prob in enumerate(dist):
            row[f'Topic_{k_id}'] = prob
        rows.append(row)
    df_theta = pd.DataFrame(rows)

    df_merged = df_w_texts[['Event_Date']].merge(
        df_theta, left_index=True, right_on='Original_Index'
    )
    df_merged['Event_Date'] = pd.to_datetime(
        df_merged['Event_Date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    df_merged = df_merged.dropna(subset=['Event_Date']).set_index('Event_Date')

    topic_cols   = [f'Topic_{nid}' for nid in narrative_topic_ids]
    available    = [c for c in topic_cols if c in df_merged.columns]
    weekly_mean  = df_merged[available].resample('W').mean().fillna(0)

    # Normalise row-wise so shares sum to 100 % each week
    row_sums = weekly_mean.sum(axis=1).replace(0, np.nan)
    weekly_norm = (weekly_mean.div(row_sums, axis=0) * 100).fillna(0)

    # Order topics by mean share descending — dominant topics at the bottom
    mean_shares  = weekly_norm.mean()
    ordered_cols = mean_shares.sort_values(ascending=False).index.tolist()
    ordered_nids = [int(c.split('_')[1]) for c in ordered_cols]

    # Build a colour palette large enough for all topics
    n = len(ordered_cols)
    cmap = plt.cm.get_cmap('tab20', max(n, 20))
    colors = [cmap(i % 20) for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(
        weekly_norm.index,
        [weekly_norm[c].values for c in ordered_cols],
        labels=[topic_id_to_name.get(nid, f'Topic {nid}') for nid in ordered_nids],
        colors=colors,
        alpha=0.85,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of narrative attention (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(
        f'Narrative composition over time — {country_name}',
        fontsize=16, fontweight='bold', pad=14,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    _draw_election_lines(ax, election_dates)

    ax.legend(
        title='Topics',
        fontsize=9,
        title_fontsize=10,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(n, 6),
        frameon=True,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    plot_path = os.path.join(output_dir, f'narrative_stacked_area_{country_name}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Stacked area plot saved to {plot_path}')


def plot_all_topics_stacked_area(
    mdl,
    df_w_texts,
    topic_id_to_name,
    output_dir="output",
    country_name="all",
    election_dates=None,
    top_n=15,
):
    """
    Stacked area chart of ALL K topics (seeded + unseeded) normalised to 100%.

    Topics are ranked by their mean weekly share. The top `top_n` are shown
    individually; all remaining lower-prominence topics are merged into a single
    grey "Other" layer at the top of the stack. This keeps the chart readable
    regardless of K while still accounting for the full probability mass.

    Seeded topics (those present in topic_id_to_name) are shown with a bold
    label in the legend so they can be distinguished from unseeded topics at a
    glance.

    top_n : int
        Number of individual topic layers to show. Remaining topics are merged
        into "Other". Set to a value >= K to show every topic individually.
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_indices = set(df_w_texts.index)
    rows = []
    for idx, doc in enumerate(mdl.docs):
        if idx not in valid_indices:
            continue
        try:
            dist = doc.get_topic_dist()
        except Exception:
            continue
        row = {'Original_Index': idx}
        for k_id, prob in enumerate(dist):
            row[f'Topic_{k_id}'] = prob
        rows.append(row)
    df_theta = pd.DataFrame(rows)

    df_merged = df_w_texts[['Event_Date']].merge(
        df_theta, left_index=True, right_on='Original_Index'
    )
    df_merged['Event_Date'] = pd.to_datetime(
        df_merged['Event_Date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    df_merged = df_merged.dropna(subset=['Event_Date']).set_index('Event_Date')

    all_cols = [f'Topic_{k}' for k in range(mdl.k) if f'Topic_{k}' in df_merged.columns]
    weekly_mean = df_merged[all_cols].resample('W').mean().fillna(0)

    # Normalise to 100% each week across ALL topics
    row_sums = weekly_mean.sum(axis=1).replace(0, np.nan)
    weekly_norm = (weekly_mean.div(row_sums, axis=0) * 100).fillna(0)

    # Order all topics by mean share descending
    mean_shares = weekly_norm.mean().sort_values(ascending=False)
    top_cols = mean_shares.index[:top_n].tolist()
    other_cols = mean_shares.index[top_n:].tolist()

    ordered_nids = [int(c.split('_')[1]) for c in top_cols]

    # Seeded topic ids for bold labelling
    seeded_ids = set(topic_id_to_name.keys())

    def _label(nid):
        if nid in seeded_ids:
            return f'[{topic_id_to_name[nid]}]'
        top_words = [w for w, _ in mdl.get_topic_words(nid, top_n=2)]
        return ', '.join(top_words)

    labels = [_label(nid) for nid in ordered_nids]

    # Colour: seeded topics get distinct tab10 colours, unseeded get tab20b
    cmap_seeded = plt.cm.get_cmap('tab10')
    cmap_unseeded = plt.cm.get_cmap('tab20b')
    seeded_counter, unseeded_counter = 0, 0
    colors = []
    for nid in ordered_nids:
        if nid in seeded_ids:
            colors.append(cmap_seeded(seeded_counter % 10))
            seeded_counter += 1
        else:
            colors.append(cmap_unseeded(unseeded_counter % 20))
            unseeded_counter += 1

    # Stack data: top topics individually, rest merged into "Other"
    stack_data = [weekly_norm[c].values for c in top_cols]
    stack_labels = labels[:]
    stack_colors = colors[:]

    if other_cols:
        other_share = weekly_norm[other_cols].sum(axis=1).values
        stack_data.append(other_share)
        stack_labels.append(f'Other ({len(other_cols)} topics)')
        stack_colors.append('#cccccc')

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(
        weekly_norm.index,
        stack_data,
        labels=stack_labels,
        colors=stack_colors,
        alpha=0.85,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of all topic attention (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(
        f'Full topic composition over time — {country_name}\n'
        f'(top {top_n} of {mdl.k} topics shown individually; seeded topics in brackets)',
        fontsize=14, fontweight='bold', pad=14,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    _draw_election_lines(ax, election_dates)

    n_legend = len(stack_labels)
    ax.legend(
        title='Topics  ([ ] = seeded)',
        fontsize=8,
        title_fontsize=9,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(n_legend, 5),
        frameon=True,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    plot_path = os.path.join(output_dir, f'all_topics_stacked_area_{country_name}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Full topic stacked area plot saved to {plot_path}')