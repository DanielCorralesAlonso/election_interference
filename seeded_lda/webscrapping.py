import os
import pandas as pd
from newspaper import Article
from tqdm import tqdm


def webscrape_articles(df, N=None, random_state=42, cache_file="gdelt/articles_with_texts.csv", use_cached_df=True, save_cache=True):
    print("\n" + "=" * 50)
    print(" Date extraction and context retrieval from news articles")
    print("=" * 50)

    if use_cached_df and os.path.exists(cache_file):
        print(f"Loading cached dataframe from: {cache_file}")
        df_w_texts = pd.read_csv(cache_file)
        print(f"Loaded {len(df_w_texts)} rows from cache.")
        return df_w_texts
    else:
        print("No cache found (or caching disabled). Starting scraping...")

        available_urls = df['News_URL'].dropna().unique().tolist()
        sample_n = min(N, len(available_urls)) if N is not None else len(available_urls)
        urls = pd.Series(available_urls).sample(n=sample_n, random_state=random_state, replace = False).tolist()

        rows = []

        for url in tqdm(urls, desc="Scraping articles", unit="article"):
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text

                event_date = df.loc[df['News_URL'] == url, 'Event_Date'].iloc[0]
                rows.append({
                    'Event_Date': event_date,
                    'News_URL': url,
                    'Full_Text': text
                })

            except Exception:
                # Keep running even if one URL fails
                continue

        df_w_texts = pd.DataFrame(rows, columns=['Event_Date', 'News_URL', 'Full_Text'])
        print(f"Scraped {len(df_w_texts)} articles successfully.")

        if save_cache:
            df_w_texts.to_csv(cache_file, index=False)
            print(f"Saved cache to: {cache_file}")

        return df_w_texts