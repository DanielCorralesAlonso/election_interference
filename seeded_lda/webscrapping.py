import csv
import json
import os
import time
import traceback
from datetime import datetime, timezone

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

# Cookie/GDPR consent walls return 200 with boilerplate text instead of the
# article (common on Yahoo etc. when the request looks like it's from the EU).
# Multi-language keyword check — any hit marks the text as a consent wall.
_CONSENT_WALL_PATTERNS = (
    'tu privacidad es importante',
    'we care about your privacy',
    'we and our partners',
    'nosotros y nuestros socios',
    'transparency and consent framework',
    'marco de transparencia y consentimiento',
    'gestionar configuración de privacidad',
    'manage privacy settings',
    'before you continue to',
    'consentement',
    'wir und unsere partner',
)


def _is_consent_wall(text):
    lowered = text.lower()
    return any(p in lowered for p in _CONSENT_WALL_PATTERNS)


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


# Common consent-management-platform buttons, tried in order. "Reject" variants
# come first so we don't opt the (fake) browser into tracking unnecessarily.
_CONSENT_BUTTON_SELECTORS = (
    '#onetrust-reject-all-handler',
    '#onetrust-accept-btn-handler',
    '#didomi-notice-disagree-button',
    '#didomi-notice-agree-button',
    '.qc-cmp2-summary-buttons button[mode="secondary"]',
    '.qc-cmp2-summary-buttons button[mode="primary"]',
    'button:has-text("Reject all")',
    'button:has-text("Accept all")',
    'button:has-text("I agree")',
    'button:has-text("Rechazar todo")',
    'button:has-text("Aceptar todo")',
)


def _dismiss_consent_banners(page):
    """Best-effort click on a CMP reject/accept button, if one is present."""
    for selector in _CONSENT_BUTTON_SELECTORS:
        try:
            page.click(selector, timeout=1500)
            page.wait_for_timeout(1000)
            return True
        except Exception:
            continue
    return False


def _fetch_playwright(url):
    """Headless Chromium fetch — bypasses JS-based bot detection and
    auto-dismisses cookie/GDPR consent banners.

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
        if _dismiss_consent_banners(page):
            page.wait_for_load_state('networkidle', timeout=10000)
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
            if _is_consent_wall(text):
                raise ValueError("consent wall detected")
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

            # --- consent wall: try Playwright (auto-dismiss banner) first,
            # then fall back to a Wayback snapshot ---
            if 'consent wall' in msg:
                tqdm.write(f"  consent wall — trying Playwright: {url}")
                stages_tried.append('playwright')
                try:
                    html = _fetch_playwright(url)
                    text = _extract(html, url)
                    if not _is_consent_wall(text) and len(text) >= _MIN_TEXT_LEN:
                        return text, 'playwright'
                    last_exc = ValueError("Playwright: still consent wall or too short")
                except Exception as pw_exc:
                    tqdm.write(f"  Playwright failed: {pw_exc}")
                    last_exc = pw_exc

                archive = _wayback_url(url)
                if archive:
                    tqdm.write(f"  consent wall — trying Wayback: {archive}")
                    stages_tried.append('wayback')
                    try:
                        html = _fetch_requests(archive)
                        text = _extract(html, archive)
                        if not _is_consent_wall(text) and len(text) >= _MIN_TEXT_LEN:
                            return text, 'wayback'
                        last_exc = ValueError("Wayback: still consent wall or too short")
                    except Exception as wb_exc:
                        last_exc = wb_exc
                else:
                    tqdm.write(f"  consent wall — no Wayback snapshot: {url}")
                break

            # --- 403 or too-short text: try Playwright ---
            if '403' in msg or 'extracted only' in msg:
                tqdm.write(f"  403/JS — trying Playwright: {url}")
                stages_tried.append('playwright')
                try:
                    html = _fetch_playwright(url)
                    text = _extract(html, url)
                    if _is_consent_wall(text):
                        last_exc = ValueError("Playwright: consent wall detected")
                    elif len(text) >= _MIN_TEXT_LEN:
                        return text, 'playwright'
                    else:
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
                        if not _is_consent_wall(text) and len(text) >= _MIN_TEXT_LEN:
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
                       error_log_file=None, partial_file=None,
                       diagnostics_file=None,
                       request_delay=0.5):
    """Scrape full article text for each URL in df.

    Uses a three-stage pipeline (requests+trafilatura → Playwright → Wayback)
    to maximise recovery. request_delay (seconds) throttles requests to reduce
    429 rate-limiting.

    File contract
    -------------
    cache_file       : the COMPLETE prescraped cache. Read only when
                       use_cached_df=True and it exists. Written only when a
                       scrape run finishes processing every URL — an interrupted
                       run never overwrites it.
    partial_file     : the in-progress scrape result. Written and flushed
                       row-by-row during scraping, so an interrupted run
                       (Ctrl+C, crash, timeout) keeps everything scraped so far.
                       Removed automatically once the run completes and
                       cache_file is written. Defaults to <cache_file>_partial.csv.
    diagnostics_file : a JSON counter of successes/failures, updated and
                       flushed after every URL so it can be inspected while
                       the run is in progress. Defaults to
                       <cache_file>_diagnostics.json.
    """
    print("\n" + "=" * 50)
    print(" Date extraction and context retrieval from news articles")
    print("=" * 50)

    if use_cached_df and os.path.exists(cache_file):
        print(f"Loading complete prescraped cache from: {cache_file}")
        df_w_texts = pd.read_csv(cache_file)
        print(f"Loaded {len(df_w_texts)} rows from cache.")
        return df_w_texts

    print("No complete prescraped cache found (or caching disabled). Starting scraping...")

    base = os.path.splitext(cache_file)[0]
    if error_log_file is None:
        error_log_file = f"{base}_errors.csv"
    if partial_file is None:
        partial_file = f"{base}_partial.csv"
    if diagnostics_file is None:
        diagnostics_file = f"{base}_diagnostics.json"

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
    cache_columns = ['Event_Date', 'News_URL', 'Full_Text', 'Country', 'fetch_source']
    completed = False

    diagnostics = {
        'total_urls': len(urls),
        'processed': 0,
        'successes': 0,
        'failures': 0,
        'success_by_source': {'requests': 0, 'playwright': 0, 'wayback': 0},
        'failure_by_type': {},
        'completed': False,
    }

    def _write_diagnostics():
        with open(diagnostics_file, 'w', encoding='utf-8') as fh:
            json.dump(diagnostics, fh, indent=2)

    # The partial file and the error log are written and flushed row-by-row,
    # so an interrupted run (Ctrl+C, crash, timeout) keeps everything scraped
    # so far instead of losing it until a final to_csv() that never happens.
    partial_fh = open(partial_file, 'w', newline='', encoding='utf-8')
    try:
        partial_writer = csv.writer(partial_fh)
        partial_writer.writerow(cache_columns)
        partial_fh.flush()

        with open(error_log_file, 'w', newline='', encoding='utf-8') as err_fh:
            err_writer = csv.writer(err_fh)
            err_writer.writerow(['timestamp', 'url', 'stages_tried', 'error_type', 'error_message'])
            err_fh.flush()

            for url in tqdm(urls, desc="Scraping articles", unit="article"):
                try:
                    text, source = _fetch_text(url)

                    event_date = df.loc[df['News_URL'] == url, 'Event_Date'].iloc[0]
                    country    = df.loc[df['News_URL'] == url, 'Initiator_Country'].iloc[0]
                    row = {
                        'Event_Date':   event_date,
                        'News_URL':     url,
                        'Full_Text':    text,
                        'Country':      country,
                        'fetch_source': source,
                    }
                    rows.append(row)

                    partial_writer.writerow([row[c] for c in cache_columns])
                    partial_fh.flush()

                    diagnostics['successes'] += 1
                    diagnostics['success_by_source'][source] = (
                        diagnostics['success_by_source'].get(source, 0) + 1
                    )

                except Exception as exc:
                    n_errors += 1
                    stages = getattr(exc, 'stages_tried', [])
                    err_writer.writerow([
                        datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        url,
                        ' → '.join(stages) if stages else 'requests',
                        type(exc).__name__,
                        traceback.format_exc().splitlines()[-1],
                    ])
                    err_fh.flush()

                    diagnostics['failures'] += 1
                    err_type = type(exc).__name__
                    diagnostics['failure_by_type'][err_type] = (
                        diagnostics['failure_by_type'].get(err_type, 0) + 1
                    )

                diagnostics['processed'] += 1
                _write_diagnostics()

                time.sleep(request_delay)

        completed = True

    except KeyboardInterrupt:
        tqdm.write(f"\nScraping interrupted by user after {len(rows)}/{len(urls)} URLs.")
    finally:
        partial_fh.close()

    diagnostics['completed'] = completed
    _write_diagnostics()

    df_w_texts = pd.DataFrame(rows, columns=cache_columns)
    print(f"Scraped {len(df_w_texts)} articles successfully. "
          f"Errors: {n_errors} (see {error_log_file})")
    print(f"Diagnostics written to: {diagnostics_file}")

    if completed and save_cache:
        df_w_texts.to_csv(cache_file, index=False)
        os.remove(partial_file)
        print(f"Scrape finished — updated complete prescraped cache: {cache_file}")
    else:
        print(f"Scrape did not finish — prescraped cache left untouched: {cache_file}")
        print(f"Partial results ({len(df_w_texts)} articles) saved to: {partial_file}")

    return df_w_texts
