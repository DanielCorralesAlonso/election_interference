import csv
import os
import time
import traceback
from datetime import datetime

import requests
import trafilatura
import pandas as pd
from tqdm import tqdm

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    'User-Agent':      _USER_AGENT,
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

# Minimum characters for extracted text to be considered a real article.
_MIN_TEXT_LEN = 150


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _extract(html, url=''):
    """Run trafilatura on raw HTML, return text string or ''."""
    text = trafilatura.extract(
        html, url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,       # prefer more text over precision
        no_fallback=False,       # use fallback extractors when main fails
    )
    return (text or '').strip()


def _fetch_requests(url):
    """Plain HTTP fetch. Returns HTML string; raises on non-2xx."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _fetch_playwright(url):
    """Headless Chromium fetch — bypasses JS-based bot detection.

    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "playwright not installed — run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=_USER_AGENT)
        page = ctx.new_page()
        page.goto(url, wait_until='networkidle', timeout=30000)
        html = page.content()
        browser.close()
    return html


def _wayback_url(url, timeout=10):
    """Return the nearest Wayback Machine snapshot URL for *url*, or None."""
    try:
        resp = requests.get(
            'https://archive.org/wayback/available',
            params={'url': url},
            timeout=timeout,
            headers=_HEADERS,
        )
        snapshot = resp.json().get('archived_snapshots', {}).get('closest', {})
        if snapshot.get('status') == '200' and snapshot.get('available'):
            return snapshot['url']
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main fetch function — three-stage pipeline
# ---------------------------------------------------------------------------

def _fetch_text(url, retries=3, backoff_base=10):
    """Return (text, source) for *url* using a three-stage fallback pipeline.

    Stage 1 — requests + trafilatura  (fast; good header control)
    Stage 2 — Playwright              (headless browser; handles JS / 403)
    Stage 3 — Wayback Machine         (archive; handles 404 / 451)

    source is one of: 'requests', 'playwright', 'wayback'.
    Raises the last exception (with stages_tried attached) when all fail.
    """
    last_exc = RuntimeError(f"All fetch stages failed for: {url}")
    stages_tried = []

    for attempt in range(retries):
        try:
            stages_tried.append('requests')
            html = _fetch_requests(url)
            text = _extract(html, url)
            if len(text) >= _MIN_TEXT_LEN:
                return text, 'requests'
            # Fetched OK but too little text — likely JS-rendered; fall through
            raise ValueError(f"trafilatura extracted only {len(text)} chars")

        except Exception as exc:
            last_exc = exc
            msg = str(exc)

            # --- 429: rate-limited — wait and retry ---
            if '429' in msg:
                wait = backoff_base * (2 ** attempt)
                tqdm.write(f"  429 (attempt {attempt+1}/{retries}) — waiting {wait}s: {url}")
                time.sleep(wait)
                continue

            # --- 403 or too-short text: try Playwright ---
            if '403' in msg or 'extracted only' in msg:
                tqdm.write(f"  403/JS — trying Playwright: {url}")
                stages_tried.append('playwright')
                try:
                    html = _fetch_playwright(url)
                    text = _extract(html, url)
                    if len(text) >= _MIN_TEXT_LEN:
                        return text, 'playwright'
                    last_exc = ValueError(f"Playwright: only {len(text)} chars extracted")
                    tqdm.write(f"  Playwright: too little text extracted")
                except Exception as pw_exc:
                    tqdm.write(f"  Playwright failed: {pw_exc}")
                    last_exc = pw_exc
                break

            # --- 404 / 451: try Wayback Machine archive ---
            if '404' in msg or '451' in msg:
                archive = _wayback_url(url)
                if archive:
                    tqdm.write(f"  {msg[:3]} — trying Wayback: {archive}")
                    stages_tried.append('wayback')
                    try:
                        html = _fetch_requests(archive)
                        text = _extract(html, archive)
                        if len(text) >= _MIN_TEXT_LEN:
                            return text, 'wayback'
                        last_exc = ValueError(f"Wayback: only {len(text)} chars extracted")
                    except Exception as wb_exc:
                        last_exc = wb_exc
                break

            # --- anything else (parse error, timeout, …) ---
            break

    last_exc.stages_tried = stages_tried
    raise last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def webscrape_articles(df, N=None, random_state=42,
                       cache_file="gdelt/articles_with_texts.csv",
                       use_cached_df=True, save_cache=True,
                       error_log_file=None, request_delay=0.5):
    """Scrape full article text for each URL in df.

    Uses a three-stage pipeline (requests+trafilatura → Playwright → Wayback)
    to maximise recovery. request_delay (seconds) throttles requests to reduce
    429 rate-limiting.
    """
    print("\n" + "=" * 50)
    print(" Date extraction and context retrieval from news articles")
    print("=" * 50)

    if use_cached_df and os.path.exists(cache_file):
        print(f"Loading cached dataframe from: {cache_file}")
        df_w_texts = pd.read_csv(cache_file)
        print(f"Loaded {len(df_w_texts)} rows from cache.")
        return df_w_texts

    print("No cache found (or caching disabled). Starting scraping...")

    if error_log_file is None:
        base = os.path.splitext(cache_file)[0]
        error_log_file = f"{base}_errors.csv"

    log_dir = os.path.dirname(error_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    available_urls = df['News_URL'].dropna().unique().tolist()
    sample_n = min(N, len(available_urls)) if N is not None else len(available_urls)
    urls = (pd.Series(available_urls)
              .sample(n=sample_n, random_state=random_state, replace=False)
              .tolist())

    rows = []
    n_errors = 0

    with open(error_log_file, 'w', newline='', encoding='utf-8') as err_fh:
        err_writer = csv.writer(err_fh)
        err_writer.writerow(['timestamp', 'url', 'stages_tried', 'error_type', 'error_message'])
        err_fh.flush()

        for url in tqdm(urls, desc="Scraping articles", unit="article"):
            try:
                text, source = _fetch_text(url)

                event_date = df.loc[df['News_URL'] == url, 'Event_Date'].iloc[0]
                country    = df.loc[df['News_URL'] == url, 'Initiator_Country'].iloc[0]
                rows.append({
                    'Event_Date':   event_date,
                    'News_URL':     url,
                    'Full_Text':    text,
                    'Country':      country,
                    'fetch_source': source,
                })

            except Exception as exc:
                n_errors += 1
                stages = getattr(exc, 'stages_tried', [])
                err_writer.writerow([
                    datetime.utcnow().isoformat(timespec='seconds'),
                    url,
                    ' → '.join(stages) if stages else 'requests',
                    type(exc).__name__,
                    traceback.format_exc().splitlines()[-1],
                ])
                err_fh.flush()

            time.sleep(request_delay)

    df_w_texts = pd.DataFrame(rows, columns=['Event_Date', 'News_URL', 'Full_Text', 'Country', 'fetch_source'])
    print(f"Scraped {len(df_w_texts)} articles successfully. "
          f"Errors: {n_errors} (see {error_log_file})")

    if save_cache:
        df_w_texts.to_csv(cache_file, index=False)
        print(f"Saved cache to: {cache_file}")

    return df_w_texts
