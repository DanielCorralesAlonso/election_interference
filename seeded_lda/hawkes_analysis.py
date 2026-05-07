import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom
from datetime import timedelta


def _resolve_columns(df, date_col=None, country_col=None):
    if date_col is None:
        for candidate in ["Event_Date", "Date", "event_date", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break

    if country_col is None:
        for candidate in ["Initiator_Country", "Country", "country"]:
            if candidate in df.columns:
                country_col = candidate
                break

    if date_col is None or country_col is None:
        raise ValueError(
            "Could not find required columns for Hawkes analysis. "
            f"Found date_col={date_col}, country_col={country_col}."
        )

    return date_col, country_col


def _country_code_from_name(value):
    key = str(value).strip().lower()
    mapping = {
        "russia": "RUS",
        "rus": "RUS",
        "ru": "RUS",
        "china": "CHN",
        "chn": "CHN",
        "cn": "CHN",
        "iran": "IRN",
        "irn": "IRN",
        "ir": "IRN",
        "all": "ALL",
    }
    return mapping.get(key, str(value).strip().upper())


def _exp_branching_ratio(alpha, decay):
    return float(alpha / max(1e-10, (1.0 - np.exp(-decay))))


def _time_features(n):
    t = np.arange(n, dtype=float)
    t_scaled = (t - t.mean()) / max(1.0, t.std())
    dow = t % 7.0
    sin_week = np.sin(2.0 * np.pi * dow / 7.0)
    cos_week = np.cos(2.0 * np.pi * dow / 7.0)
    return t_scaled, sin_week, cos_week


def _baseline_intensity(n, beta0, beta_trend=0.0, beta_sin=0.0, beta_cos=0.0, baseline="constant"):
    if baseline == "constant":
        return np.exp(beta0) * np.ones(n, dtype=float)

    t_scaled, sin_week, cos_week = _time_features(n)
    eta = beta0 + beta_trend * t_scaled + beta_sin * sin_week + beta_cos * cos_week
    return np.exp(eta)


def _exp_intensity(counts, beta0, alpha, decay, beta_trend=0.0, beta_sin=0.0, beta_cos=0.0, baseline="constant"):
    n = len(counts)
    excitation = np.zeros(n, dtype=float)
    for t in range(1, n):
        excitation[t] = excitation[t - 1] * np.exp(-decay) + alpha * counts[t - 1]
    intensity = _baseline_intensity(
        n,
        beta0=beta0,
        beta_trend=beta_trend,
        beta_sin=beta_sin,
        beta_cos=beta_cos,
        baseline=baseline,
    ) + excitation
    return np.clip(intensity, 1e-10, None)


def _loglik_from_intensity(counts, intensity, likelihood, r=None):
    if likelihood == "poisson":
        return float(np.sum(counts * np.log(intensity) - intensity - gammaln(counts + 1)))

    term1 = gammaln(counts + r) - gammaln(r) - gammaln(counts + 1)
    term2 = r * np.log(r + 1e-10) - r * np.log(r + intensity + 1e-10)
    term3 = counts * np.log(intensity + 1e-10) - counts * np.log(r + intensity + 1e-10)
    return float(np.sum(term1 + term2 + term3))


def _fit_hawkes_exponential_single_likelihood(
    ts,
    likelihood="neg_binomial",
    baseline="weekly_trend",
    decay=0.8,
    fit_decay=True,
    n_starts=8,
    random_state=42,
):
    ts = np.asarray(ts, dtype=float)
    rng = np.random.default_rng(random_state)

    def unpack_params(params):
        idx = 0
        beta0 = params[idx]
        idx += 1
        beta_trend = 0.0
        beta_sin = 0.0
        beta_cos = 0.0
        if baseline == "weekly_trend":
            beta_trend = params[idx]
            idx += 1
            beta_sin = params[idx]
            idx += 1
            beta_cos = params[idx]
            idx += 1
        alpha = params[idx]
        idx += 1
        r = None
        if likelihood == "neg_binomial":
            r = params[idx]
            idx += 1
        local_decay = decay
        if fit_decay:
            local_decay = params[idx]
        return beta0, beta_trend, beta_sin, beta_cos, alpha, r, local_decay

    def objective(params):
        beta0, beta_trend, beta_sin, beta_cos, alpha, r, local_decay = unpack_params(params)
        if alpha <= 1e-8 or local_decay <= 1e-4:
            return 1e10
        if likelihood == "neg_binomial" and (r is None or r <= 1e-5):
            return 1e10

        branching = _exp_branching_ratio(alpha, local_decay)
        if branching >= 0.995:
            return 1e9 + 1e7 * (branching - 0.995)

        intensity = _exp_intensity(
            ts,
            beta0=beta0,
            alpha=alpha,
            decay=local_decay,
            beta_trend=beta_trend,
            beta_sin=beta_sin,
            beta_cos=beta_cos,
            baseline=baseline,
        )
        log_lik = _loglik_from_intensity(ts, intensity, likelihood, r=r)
        return -log_lik

    base_mu = max(float(np.mean(ts)), 1e-2)
    base_beta0 = float(np.log(base_mu + 1e-6))
    bounds = [(-8.0, 8.0)]
    if baseline == "weekly_trend":
        bounds.extend([(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)])
    bounds.append((1e-8, 10.0))
    if likelihood == "neg_binomial":
        bounds.append((1e-4, 200.0))
    if fit_decay:
        bounds.append((0.02, 5.0))

    starts = []
    for _ in range(max(1, n_starts)):
        x0 = [base_beta0 + float(rng.uniform(-0.7, 0.7))]
        if baseline == "weekly_trend":
            x0.extend([
                float(rng.uniform(-0.2, 0.2)),
                float(rng.uniform(-0.5, 0.5)),
                float(rng.uniform(-0.5, 0.5)),
            ])
        x0.append(float(rng.uniform(0.005, 0.4)))
        if likelihood == "neg_binomial":
            x0.append(float(rng.uniform(0.3, 10.0)))
        if fit_decay:
            x0.append(float(rng.uniform(0.15, 2.5)))
        starts.append(x0)

    best_res = None
    best_obj = np.inf
    for x0 in starts:
        res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
        if np.isfinite(res.fun) and res.fun < best_obj:
            best_obj = res.fun
            best_res = res

    if best_res is None:
        raise RuntimeError("Hawkes optimization failed for all initializations.")

    beta0, beta_trend, beta_sin, beta_cos, alpha, r, local_decay = unpack_params(best_res.x)
    intensity = _exp_intensity(
        ts,
        beta0=beta0,
        alpha=alpha,
        decay=local_decay,
        beta_trend=beta_trend,
        beta_sin=beta_sin,
        beta_cos=beta_cos,
        baseline=baseline,
    )
    log_lik = _loglik_from_intensity(ts, intensity, likelihood, r=r)

    if likelihood == "poisson":
        lower = poisson.ppf(0.025, intensity)
        upper = poisson.ppf(0.975, intensity)
    else:
        prob = r / (r + intensity)
        lower = nbinom.ppf(0.025, r, prob)
        upper = nbinom.ppf(0.975, r, prob)

    k = 2 + (1 if likelihood == "neg_binomial" else 0) + (1 if fit_decay else 0)
    n = max(1, len(ts))
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    branching = _exp_branching_ratio(alpha, local_decay)

    baseline_mean = float(np.mean(_baseline_intensity(
        len(ts),
        beta0=beta0,
        beta_trend=beta_trend,
        beta_sin=beta_sin,
        beta_cos=beta_cos,
        baseline=baseline,
    )))
    params = {
        "likelihood": likelihood,
        "baseline": baseline,
        "baseline_mean": round(baseline_mean, 6),
        "beta0": round(float(beta0), 6),
        "alpha": round(float(alpha), 6),
        "decay": round(float(local_decay), 6),
        "branching_ratio": round(float(branching), 6),
        "aic": round(float(aic), 4),
        "bic": round(float(bic), 4),
        "converged": bool(best_res.success),
        "optimizer_message": str(best_res.message),
        "n_starts": int(max(1, n_starts)),
    }
    if baseline == "weekly_trend":
        params["beta_trend"] = round(float(beta_trend), 6)
        params["beta_sin"] = round(float(beta_sin), 6)
        params["beta_cos"] = round(float(beta_cos), 6)
        params["weekly_amplitude"] = round(float(np.sqrt(beta_sin ** 2 + beta_cos ** 2)), 6)
    if likelihood == "neg_binomial":
        params["r"] = round(float(r), 6)

    return intensity, lower, upper, params, float(log_lik)


def fit_hawkes(
    ts,
    kernel="exponential",
    likelihood="auto",
    baseline="auto",
    decay=0.8,
    fit_decay=True,
    n_starts=8,
    random_state=42,
):
    """
    Fit Hawkes process to daily counts.

    Improvements over the basic version:
    - multi-start optimization to reduce local minima issues;
    - stationarity control through branching ratio constraint;
    - optional decay estimation (fit_decay=True);
    - likelihood='auto' model selection via BIC between Poisson and NB.
    """
    if kernel != "exponential":
        raise NotImplementedError("This implementation currently supports only the exponential kernel.")

    if baseline not in {"auto", "constant", "weekly_trend"}:
        raise ValueError("baseline must be one of: 'auto', 'constant', 'weekly_trend'.")

    candidate_baselines = [baseline] if baseline != "auto" else ["constant", "weekly_trend"]
    candidate_likelihoods = [likelihood] if likelihood != "auto" else ["poisson", "neg_binomial"]

    best_choice = None
    best_bic = np.inf
    candidates = []

    for baseline_mode in candidate_baselines:
        for likelihood_mode in candidate_likelihoods:
            result = _fit_hawkes_exponential_single_likelihood(
                ts,
                likelihood=likelihood_mode,
                baseline=baseline_mode,
                decay=decay,
                fit_decay=fit_decay,
                n_starts=n_starts,
                random_state=random_state + len(candidates) * 17,
            )
            candidates.append((baseline_mode, likelihood_mode, result))
            bic = result[3]["bic"]
            if bic < best_bic:
                best_bic = bic
                best_choice = (baseline_mode, likelihood_mode, result)

    if best_choice is None:
        raise RuntimeError("No Hawkes candidate converged.")

    chosen_baseline, chosen_likelihood, chosen = best_choice
    chosen[3]["selection"] = "auto_bic" if (baseline == "auto" or likelihood == "auto") else "manual"
    chosen[3]["chosen_baseline"] = chosen_baseline
    chosen[3]["chosen_likelihood"] = chosen_likelihood
    chosen[3]["candidates"] = [
        {"baseline": b, "likelihood": l, "bic": r[3]["bic"], "converged": r[3]["converged"]}
        for b, l, r in candidates
    ]
    if len(candidates) > 1:
        alt = sorted(candidates, key=lambda item: item[2][3]["bic"])
        chosen[3]["runner_up_bic"] = alt[1][2][3]["bic"] if len(alt) > 1 else None
    return chosen


def run_hawkes_news_analysis(
    df,
    output_dir="output",
    country_name="all",
    countries=("RUS", "CHN", "IRN"),
    date_col=None,
    country_col=None,
    likelihood="auto",
    baseline="auto",
    fit_decay=True,
    n_starts=10,
    random_state=42,
):
    os.makedirs(output_dir, exist_ok=True)
    date_col, country_col = _resolve_columns(df, date_col=date_col, country_col=country_col)

    working = df[[date_col, country_col]].dropna().copy()
    raw_dates = working[date_col].astype(str)
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    if parsed_dates.isna().all():
        parsed_dates = pd.to_datetime(raw_dates.str.replace(r"[^0-9]", "", regex=True), format="%Y%m%d", errors="coerce")
    working["_date"] = parsed_dates
    working = working.dropna(subset=["_date"])
    if working.empty:
        raise ValueError("No valid dates found for Hawkes analysis.")

    selected = []
    if _country_code_from_name(country_name) == "ALL":
        selected = list(countries)
    else:
        selected = [_country_code_from_name(country_name)]

    summary = {
        "date_col": date_col,
        "country_col": country_col,
        "countries": {},
        "analysis": {
            "likelihood": likelihood,
            "baseline": baseline,
            "fit_decay": fit_decay,
            "n_starts": n_starts,
            "random_state": random_state,
        },
    }

    for code in selected:
        c_mask = working[country_col].astype(str).str.strip().str.upper() == code
        country_df = working[c_mask]
        if country_df.empty:
            continue

        start_date = country_df["_date"].min().normalize()
        end_date = country_df["_date"].max().normalize()
        full_dates = pd.date_range(start=start_date, end=end_date, freq="D")

        daily = country_df.groupby("_date").size().rename("News")
        ts = daily.reindex(full_dates, fill_value=0).values

        intensity, lower, upper, params, log_lik = fit_hawkes(
            ts,
            kernel="exponential",
            likelihood=likelihood,
            baseline=baseline,
            decay=0.8,
            fit_decay=fit_decay,
            n_starts=n_starts,
            random_state=random_state,
        )

        span_label = f"{start_date.year}_{end_date.year}" if start_date.year != end_date.year else f"{start_date.year}"

        plt.figure(figsize=(14, 5))
        plt.plot(full_dates, ts, color="black", alpha=0.4, linewidth=2.5, label="Daily news count")
        fitted_label = f"Hawkes intensity ({params.get('likelihood', likelihood)})"
        plt.plot(full_dates, intensity, color="red", linestyle="--", linewidth=2.2, label=fitted_label)
        plt.fill_between(full_dates, lower, upper, color="red", alpha=0.2, label="95% CI")
        plt.title(f"Hawkes fit of daily news volume ({code}) - {span_label}")
        plt.xlabel("Date")
        plt.ylabel("Daily article count")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        out_plot = os.path.join(output_dir, f"hawkes_news_{code}_{span_label}.png")
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        plt.close()

        daily_frame = pd.DataFrame(
            {
                "Date": full_dates,
                "Observed_News": ts,
                "Fitted_Intensity": intensity,
                "Lower_95": lower,
                "Upper_95": upper,
            }
        )
        out_csv = os.path.join(output_dir, f"hawkes_news_{code}_{start_date.year}_{end_date.year}.csv")
        daily_frame.to_csv(out_csv, index=False)

        weekday_profile = country_df.assign(weekday=country_df["_date"].dt.day_name()).groupby("weekday").size()
        weekday_profile = weekday_profile.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).fillna(0)
        weekend_mean = float(weekday_profile.loc[["Saturday", "Sunday"]].mean())
        weekday_mean = float(weekday_profile.loc[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean())
        weekend_ratio = weekend_mean / max(1e-10, weekday_mean)

        summary["countries"][code] = {
            "n_days": int(len(ts)),
            "total_articles": int(np.sum(ts)),
            "log_likelihood": round(log_lik, 4),
            "parameters": params,
            "plot_path": out_plot,
            "series_csv": out_csv,
            "weekday_profile": {day: int(val) for day, val in weekday_profile.items()},
            "weekend_to_weekday_ratio": round(float(weekend_ratio), 6),
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
        }

    out_json = os.path.join(output_dir, f"hawkes_news_summary_{country_name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
