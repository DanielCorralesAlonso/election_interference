import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from hmmlearn.hmm import PoissonHMM
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom, norm, kstest, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import os

warnings.filterwarnings("ignore")

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

# ==========================================
# 1. EVALUATION METRICS
# ==========================================
def calc_mase(actual, predicted):
    naive_mae = np.mean(np.abs(np.diff(actual)))
    if naive_mae == 0: return np.nan
    return np.mean(np.abs(actual - predicted)) / naive_mae

def calc_mda(actual, predicted):
    actual_dir = np.sign(actual[1:] - actual[:-1])
    pred_dir = np.sign(predicted[1:] - predicted[:-1])
    valid = actual_dir != 0
    if sum(valid) == 0: return np.nan
    return np.mean(actual_dir[valid] == pred_dir[valid])

def calc_ic(log_lik, k, n):
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    return aic, bic

def extract_hawkes_catalysts(dates, actual_counts, predicted_intensity, mu_baseline, top_n=3):
    """Uses Stochastic Declustering to find the root Exogenous Shocks (Mainshocks)."""
    safe_intensity = np.maximum(predicted_intensity, 1e-5)
    organic_ratio = mu_baseline / safe_intensity
    organic_volume = actual_counts * organic_ratio
    
    df_declustered = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Total_Articles': actual_counts.astype(int),
        'Expected_Intensity': np.round(predicted_intensity, 1),
        # FIXED: Keep this as a pure float so the main execution block can format it safely
        'Organic_Ratio': organic_ratio * 100, 
        'Organic_Volume': np.round(organic_volume, 1)
    })
    
    # Filter for days above the 75th percentile of volume to find true heavy days
    vol_threshold = np.percentile(actual_counts, 75)
    df_heavy = df_declustered[df_declustered['Total_Articles'] >= vol_threshold]
    
    catalysts = df_heavy.sort_values(by='Organic_Volume', ascending=False).head(top_n).copy()
    return catalysts

# ==========================================
# 2. MODELS WITH UNCERTAINTY BANDS
# ==========================================
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom, norm
from scipy.optimize import minimize
from scipy.stats import binom, poisson, norm

def fit_discrete_ar(ts, lags=1, likelihood='poisson'):
    """
    Fits a Discrete Autoregressive model (Poisson or Negative Binomial).
    This outputs true discrete log-likelihoods, allowing direct AIC/BIC 
    comparisons with the discrete Hawkes processes.
    """
    n = len(ts)
    
    def obj_func(params, counts):
        if likelihood == 'poisson':
            mu = params[0]
            alphas = params[1:]
        else:
            mu = params[0]
            alphas = params[1:-1]
            r = params[-1]
            if r <= 1e-5: return 1e9
            
        if mu <= 1e-5 or any(a <= 1e-5 for a in alphas) or sum(alphas) >= 1.0: 
            return 1e9 # Prevent negative intensities or explosive non-stationary processes
        
        intensity = np.full(n, mu)
        
        # Calculate AR intensity
        for t in range(lags, n):
            for i in range(lags):
                intensity[t] += alphas[i] * counts[t - 1 - i]
                
        log_lik = 0
        # Only evaluate likelihood AFTER the warmup period (t >= lags)
        for t in range(lags, n):
            if likelihood == 'poisson':
                log_lik += counts[t] * np.log(intensity[t] + 1e-10) - intensity[t] - gammaln(counts[t] + 1)
            else:
                term1 = gammaln(counts[t] + r) - gammaln(r) - gammaln(counts[t] + 1)
                term2 = r * np.log(r + 1e-10) - r * np.log(r + intensity[t] + 1e-10)
                term3 = counts[t] * np.log(intensity[t] + 1e-10) - counts[t] * np.log(r + intensity[t] + 1e-10)
                log_lik += term1 + term2 + term3
                
        return -log_lik

    # Initial guesses and bounds
    x0 = [np.mean(ts) * 0.5] + [0.1] * lags
    bounds = [(1e-2, None)] + [(1e-5, 0.99)] * lags
    if likelihood == 'neg_binomial':
        x0.append(1.0)
        bounds.append((1e-2, None))

    res = minimize(obj_func, x0=x0, args=(ts,), bounds=bounds, method='L-BFGS-B')
    
    # Extract parameters
    if likelihood == 'poisson':
        mu = res.x[0]
        alphas = res.x[1:]
        params = {'mu': round(mu, 2)}
        for i, a in enumerate(alphas): params[f'lag_{i+1}'] = round(a, 3)
        k = 1 + lags
    else:
        mu = res.x[0]
        alphas = res.x[1:-1]
        r = res.x[-1]
        params = {'mu': round(mu, 2), 'r': round(r, 2)}
        for i, a in enumerate(alphas): params[f'lag_{i+1}'] = round(a, 3)
        k = 2 + lags

    # Rebuild full intensity array
    intensity = np.full(n, mu)
    for t in range(lags, n):
        for i in range(lags):
            intensity[t] += alphas[i] * ts[t - 1 - i]
            
    # Calculate 95% Confidence Intervals
    if likelihood == 'poisson':
        lower = poisson.ppf(0.025, intensity)
        upper = poisson.ppf(0.975, intensity)
    else:
        p = r / (r + intensity)
        lower = nbinom.ppf(0.025, r, p)
        upper = nbinom.ppf(0.975, r, p)
        
    return intensity, lower, upper, params, -res.fun, k

def fit_hawkes(ts, decay=0.8, likelihood='poisson'):
    def obj_func(params, counts, decay, likelihood):
        if likelihood == 'poisson':
            mu, alpha = params
        else:
            mu, alpha, r = params
            if r <= 1e-5: return 1e9 
            
        if mu <= 1e-5 or alpha <= 1e-5 or alpha >= 0.99: 
            return 1e9 
        
        n = len(counts)
        intensity = np.zeros(n)
        intensity[0] = mu
        log_lik = 0
        
        for t in range(1, n):
            intensity[t] = mu + intensity[t-1] * np.exp(-decay) + alpha * counts[t-1]
            if likelihood == 'poisson':
                log_lik += counts[t] * np.log(intensity[t] + 1e-10) - intensity[t] - gammaln(counts[t] + 1)
            else:
                term1 = gammaln(counts[t] + r) - gammaln(r) - gammaln(counts[t] + 1)
                term2 = r * np.log(r + 1e-10) - r * np.log(r + intensity[t] + 1e-10)
                term3 = counts[t] * np.log(intensity[t] + 1e-10) - counts[t] * np.log(r + intensity[t] + 1e-10)
                log_lik += term1 + term2 + term3
        return -log_lik

    x0 = [np.mean(ts), 0.1]
    bounds = [(1e-2, None), (1e-5, 0.99)]
    if likelihood == 'neg_binomial':
        x0.append(1.0)
        bounds.append((1e-2, None))

    res = minimize(obj_func, x0=x0, args=(ts, decay, likelihood), bounds=bounds, method='L-BFGS-B')
    
    if likelihood == 'poisson':
        mu, alpha = res.x
        params = {'mu': round(mu, 2), 'alpha': round(alpha, 3)}
        k = 2
    else:
        mu, alpha, r = res.x
        params = {'mu': round(mu, 2), 'alpha': round(alpha, 3), 'r': round(r, 2)}
        k = 3
        
    intensity = np.zeros(len(ts))
    intensity[0] = mu
    for t in range(1, len(ts)):
         intensity[t] = mu + intensity[t-1] * np.exp(-decay) + alpha * ts[t-1]
            
    # Calculate 95% Confidence Intervals based on the chosen distribution
    if likelihood == 'poisson':
        lower = poisson.ppf(0.025, intensity)
        upper = poisson.ppf(0.975, intensity)
    else:
        # SciPy's nbinom parameterization: n = r, p = r / (r + mean)
        p = r / (r + intensity)
        lower = nbinom.ppf(0.025, r, p)
        upper = nbinom.ppf(0.975, r, p)
            
    return intensity, lower, upper, params, -res.fun, k

def fit_hmm(ts, n_components=2):
    X = ts.reshape(-1, 1)
    model = PoissonHMM(n_components=n_components, n_iter=200, random_state=42).fit(X)
    
    lambdas = np.round(model.lambdas_.flatten(), 1)
    trans_mat = np.round(model.transmat_, 2)
    params = {'L_0': lambdas[0], 'L_1': lambdas[1], 'P(0->0)': trans_mat[0,0], 'P(1->1)': trans_mat[1,1]}
    k = n_components + (n_components * (n_components - 1))
    
    hidden_states = model.predict(X)
    preds = np.array([model.lambdas_.flatten()[state] for state in hidden_states])
    
    # 95% Confidence Intervals for the chosen Poisson state
    lower = poisson.ppf(0.025, preds)
    upper = poisson.ppf(0.975, preds)
    
    return preds, lower, upper, params, model.score(X), k

def fit_poisson_process(ts):
    """
    Fits a standard Homogeneous Poisson Process.
    Assumption: Constant baseline intensity, no self-excitation, variance = mean.
    """
    mu = np.mean(ts)
    mu = max(mu, 1e-5) # Prevent log(0)
    
    # Log-likelihood for constant Poisson
    log_lik = np.sum(ts * np.log(mu) - mu - gammaln(ts + 1))
    
    intensity = np.full(len(ts), mu)
    lower = poisson.ppf(0.025, intensity)
    upper = poisson.ppf(0.975, intensity)
    
    params = {'mu': round(mu, 2)}
    
    return intensity, lower, upper, params, log_lik, 1 # k=1 parameter

def fit_inhomogeneous_poisson(ts):
    """
    Fits an Inhomogeneous Poisson Process with a linear trend and weekly seasonality.
    Assumption: News volume changes based on the calendar/weekdays, but has no contagion.
    """
    n = len(ts)
    t_idx = np.arange(n) # Time index 0 to n-1
    
    def obj_func(params, counts):
        b0, b_trend, b_sin, b_cos = params
        
        # Calculate intensity using an exponential link function to guarantee it stays positive
        # b0: base rate, b_trend: long-term slope, b_sin/b_cos: 7-day weekly waves
        intensity = np.exp(b0 + b_trend * t_idx + 
                           b_sin * np.sin(2 * np.pi * t_idx / 7) + 
                           b_cos * np.cos(2 * np.pi * t_idx / 7))
        
        # Clip to prevent math overflow if the optimizer tests crazy parameters
        intensity = np.clip(intensity, 1e-5, 1e5)
        
        # Standard Poisson Log-Likelihood
        log_lik = np.sum(counts * np.log(intensity) - intensity - gammaln(counts + 1))
        return -log_lik

    # Initial guesses: Intercept is log(mean), zero trend, zero seasonality
    x0 = [np.log(np.mean(ts) + 1e-5), 0.0, 0.0, 0.0]
    
    # Optimize
    res = minimize(obj_func, x0=x0, args=(ts,), method='L-BFGS-B')
    
    # Extract parameters and rebuild the final wavy intensity
    b0, b_trend, b_sin, b_cos = res.x
    intensity = np.exp(b0 + b_trend * t_idx + 
                       b_sin * np.sin(2 * np.pi * t_idx / 7) + 
                       b_cos * np.cos(2 * np.pi * t_idx / 7))
    
    # Calculate Confidence Intervals
    lower = poisson.ppf(0.025, intensity)
    upper = poisson.ppf(0.975, intensity)
    
    params = {'base': round(b0, 2), 'trend': round(b_trend, 4), 'wave_amp': round(np.sqrt(b_sin**2 + b_cos**2), 2)}
    
    return intensity, lower, upper, params, -res.fun, 4 # 4 parameters

def fit_hawkes_powerlaw(ts, likelihood='neg_binomial'):
    """
    Fits a Hawkes Process with a Power Law decay kernel.
    Assumption: Events have 'long memory' and decay following alpha / lag^p.
    """
    n = len(ts)
    
    def obj_func(params, counts):
        mu, alpha, p_decay, r = params
        if mu <= 1e-5 or alpha <= 1e-5 or p_decay <= 1.01 or r <= 1e-5: return 1e9 
        
        intensity = np.zeros(n)
        intensity[0] = mu
        log_lik = 0
        
        for t in range(1, n):
            # Calculate the power law kernel for all previous days (lags 1 to t)
            lags = np.arange(1, t + 1)
            kernel = alpha / (lags ** p_decay)
            
            # Sum the influence of all historical events (counts[t-1] down to counts[0])
            intensity[t] = mu + np.sum(kernel * counts[t-1::-1])
            
            # Negative Binomial Log-Likelihood
            term1 = gammaln(counts[t] + r) - gammaln(r) - gammaln(counts[t] + 1)
            term2 = r * np.log(r + 1e-10) - r * np.log(r + intensity[t] + 1e-10)
            term3 = counts[t] * np.log(intensity[t] + 1e-10) - counts[t] * np.log(r + intensity[t] + 1e-10)
            log_lik += term1 + term2 + term3
            
        return -log_lik

    # Initial guesses: mu, alpha, p_decay, r
    # p_decay is constrained > 1.01 to ensure the process is mathematically stationary (doesn't explode to infinity)
    x0 = [np.mean(ts)*0.5, 0.1, 1.5, 1.0]
    bounds = [(1e-2, None), (1e-5, None), (1.01, 10), (1e-2, None)]
    
    res = minimize(obj_func, x0=x0, args=(ts,), bounds=bounds, method='L-BFGS-B')
    
    mu, alpha, p_decay, r = res.x
    params = {'mu': round(mu, 2), 'alpha': round(alpha, 3), 'p': round(p_decay, 2), 'r': round(r, 2)}
    
    # Rebuild intensity for output
    intensity = np.zeros(n)
    intensity[0] = mu
    for t in range(1, n):
        lags = np.arange(1, t + 1)
        kernel = alpha / (lags ** p_decay)
        intensity[t] = mu + np.sum(kernel * ts[t-1::-1])
        
    p_prob = r / (r + intensity)
    lower = nbinom.ppf(0.025, r, p_prob)
    upper = nbinom.ppf(0.975, r, p_prob)
    
    return intensity, lower, upper, params, -res.fun, 4

def fit_inar(ts, lags=1):
    """
    Fits an Integer-Valued Autoregressive (INAR) model.
    Currently restricted to INAR(1) to allow for exact Log-Likelihood 
    computation, which is required for fair AIC/BIC comparisons.
    """
    if lags > 1:
        raise NotImplementedError(
            "Exact Maximum Likelihood for INAR(p) with p > 1 is computationally "
            "intractable in simple forms due to multidimensional convolutions. "
            "Please use lags=1 for an exact likelihood comparison."
        )
        
    n = len(ts)
    
    def obj_func(params, counts):
        alpha, mu = params
        
        # alpha is the survival probability (0 to 1)
        # mu is the rate of new Poisson arrivals (> 0)
        if alpha <= 1e-5 or alpha >= 0.999 or mu <= 1e-5: 
            return 1e9
        
        log_lik = 0
        
        # Compute exact log-likelihood using the convolution of Binomial and Poisson
        for t in range(1, n):
            y_curr = int(counts[t])
            y_prev = int(counts[t-1])
            
            # The maximum number of events that could have survived from yesterday
            max_survivors = min(y_curr, y_prev)
            i_vals = np.arange(max_survivors + 1)
            
            # P(i survivors | y_prev events, alpha probability)
            prob_surv = binom.pmf(i_vals, y_prev, alpha)
            
            # P(y_curr - i new arrivals | mu rate)
            prob_inno = poisson.pmf(y_curr - i_vals, mu)
            
            # Sum the probabilities of all valid combinations
            prob_t = np.sum(prob_surv * prob_inno)
            
            # Catch impossible states to prevent log(0)
            if prob_t <= 0:
                return 1e9
                
            log_lik += np.log(prob_t)
            
        return -log_lik

    # Initial guesses: assume 50% survival, and the rest made up by new arrivals
    x0 = [0.5, np.mean(ts) * 0.5]
    bounds = [(1e-5, 0.999), (1e-5, None)]
    
    res = minimize(obj_func, x0=x0, args=(ts,), bounds=bounds, method='L-BFGS-B')
    
    alpha, mu = res.x
    params = {'alpha': round(alpha, 3), 'mu': round(mu, 2)}
    
    # Calculate Conditional Expectation (Intensity)
    # E[y_t | y_{t-1}] = alpha * y_{t-1} + mu
    intensity = np.zeros(n)
    intensity[0] = mu / (1 - alpha)  # Unconditional mean for the first step
    for t in range(1, n):
        intensity[t] = alpha * ts[t-1] + mu
        
    # Calculate Conditional Variance
    # Var(y_t | y_{t-1}) = alpha * (1 - alpha) * y_{t-1} + mu
    cond_var = np.zeros(n)
    cond_var[0] = mu / (1 - alpha)
    for t in range(1, n):
        cond_var[t] = alpha * (1 - alpha) * ts[t-1] + mu
        
    # Calculate 95% Confidence Intervals
    # Because the exact inverse CDF of the convolution is extremely slow to compute,
    # we use a Gaussian approximation based on the exact conditional mean and variance.
    lower = np.maximum(0, norm.ppf(0.025, loc=intensity, scale=np.sqrt(cond_var)))
    upper = np.maximum(0, norm.ppf(0.975, loc=intensity, scale=np.sqrt(cond_var)))
    
    return intensity, lower, upper, params, -res.fun, 2 # k=2 parameters (alpha, mu)



# Calculate Randomized Quantile Residuals for discrete models and perform K-S test

def evaluate_discrete_residuals(actual, predicted_means, model_name, country_name, output_dir, likelihood='poisson', r_disp=None, lags=1, color='blue'):
    """
    Evaluates goodness-of-fit using Randomized Quantile Residuals (RQRs).
    Returns K-S stats (distribution fit) and Ljung-Box stats (autocorrelation).
    """
    y_act = actual[lags:]
    y_pred = predicted_means[lags:]
    n = len(y_act)
    
    residuals = np.zeros(n)
    u_vals = np.zeros(n) # NEW: We need to store the uniform variables for the PIT plot
    np.random.seed(42) 
    
    for t in range(n):
        y = y_act[t]
        mu = max(y_pred[t], 1e-5)
        
        if likelihood == 'poisson':
            cdf_y = poisson.cdf(y, mu)
            cdf_y_minus_1 = 0.0 if y == 0 else poisson.cdf(y - 1, mu)
        elif likelihood == 'neg_binomial':
            p = r_disp / (r_disp + mu)
            cdf_y = nbinom.cdf(y, r_disp, p)
            cdf_y_minus_1 = 0.0 if y == 0 else nbinom.cdf(y - 1, r_disp, p)
            
        u = np.random.uniform(cdf_y_minus_1, cdf_y)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        
        u_vals[t] = u            # Store Uniform values for PIT
        residuals[t] = norm.ppf(u) # Transform to Normal for KS / Ljung-Box
        
    # Run Kolmogorov-Smirnov Test
    ks_stat, ks_pval = kstest(residuals, 'norm')
    
    # NEW: Run Ljung-Box Test (Testing up to 10 days of lag for leftover autocorrelation)
    lb_res = acorr_ljungbox(residuals, lags=[7], return_df=True)
    lb_stat = lb_res['lb_stat'].iloc[0]
    lb_pval = lb_res['lb_pvalue'].iloc[0]
    
    # Generate Plots
    _plot_qq(residuals, model_name, country_name, ks_stat, ks_pval, color, output_dir)
    _plot_pit(u_vals, model_name, country_name, color, output_dir) # NEW: PIT Plot
    
    return ks_stat, ks_pval, lb_stat, lb_pval

def _plot_pit(u_vals, model_name, country_name, color, output_dir):
    """
    Plots a Probability Integral Transform (PIT) Histogram.
    A perfectly calibrated model will have a flat, uniform histogram.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot histogram of the uniform U values
    counts, bins, patches = ax.hist(u_vals, bins=10, density=True, color=color, alpha=0.6, edgecolor='black')
    
    # Add a horizontal line at y=1 (the theoretical ideal for a Uniform distribution)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Ideal Uniform')
    
    ax.set_title(f'{model_name} PIT Histogram: {country_name}\n(Flat = Perfectly Calibrated Uncertainty)')
    ax.set_xlabel('Probability Integral Transform (u)')
    ax.set_ylabel('Density')
    ax.set_ylim(0, max(1.5, max(counts) * 1.2)) # Give it some headroom
    ax.legend()
    
    plt.tight_layout()
    filename = f"{country_name}_{model_name.replace(' ', '_')}_PIT.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

def _plot_qq(residuals, model_name, country_name, ks_stat, p_val, color, output_dir):
    """
    Helper function to plot and save Q-Q plots for the residuals.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    probplot(residuals, dist="norm", plot=ax)
    
    # Styling
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor(color)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linestyle('--')
    
    ax.set_title(f'{model_name} Q-Q: {country_name}\nK-S: {ks_stat:.3f} | p-val: {p_val:.3e}')
    plt.tight_layout()
    
    # Ensure directory exists before saving
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{country_name}_{model_name.replace(' ', '_')}_QQ.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

# ==========================================
# 3. MAIN EXECUTION & FORMATTING
# ==========================================
if __name__ == "__main__":
    output_dir = "model_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to the '{output_dir}' directory.")

    # Load your actual GDELT data
    df = pd.read_csv('data/gdelt/daily_news_data.csv')

    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    for country in ['RUS', 'CHN', 'IRN']:
        print(f"\n{'='*75}\n Analyzing Country: {country}\n{'='*75}")
        
        country_data = df[df['Initiator_Country'] == country].copy()
        country_data['Date'] = pd.to_datetime(country_data['Date'])
        ts_series = country_data.set_index('Date')['News']
        
        full_dates = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
        ts = ts_series.reindex(full_dates, fill_value=0).values
        n = len(ts)
        
        if np.sum(ts) == 0:
            print(f"No data found for {country}. Skipping.")
            continue
            
        # ==========================================
        # 1. Fit Models (All operating in Discrete Probability Space)
        # ==========================================
        pp_p, pp_l, pp_u, pp_params, pp_ll, pp_k = fit_poisson_process(ts) 
        ihp_p, ihp_l, ihp_u, ihp_params, ihp_ll, ihp_k = fit_inhomogeneous_poisson(ts) 
        ar_p, ar_l, ar_u, ar_params, ar_ll, ar_k = fit_discrete_ar(ts, lags=7, likelihood='poisson') # UPDATED
        inar_p, inar_l, inar_u, inar_params, inar_ll, inar_k = fit_inar(ts, lags=1) # NEW
        
        hp_p, hp_l, hp_u, hp_params, hp_ll, hp_k = fit_hawkes(ts, decay=0.8, likelihood='poisson')
        hnb_p, hnb_l, hnb_u, hnb_params, hnb_ll, hnb_k = fit_hawkes(ts, decay=0.8, likelihood='neg_binomial')
        hpow_p, hpow_l, hpow_u, hpow_params, hpow_ll, hpow_k = fit_hawkes_powerlaw(ts, likelihood='neg_binomial') 
        hmm_p, hmm_l, hmm_u, hmm_params, hmm_ll, hmm_k = fit_hmm(ts, n_components=2)
        
        # ==========================================
        # 2. Evaluate Residuals & K-S Tests (Unified)
        # ==========================================
        ks_pp_stat, ks_pp_pval, lb_pp_stat, lb_pp_pval = evaluate_discrete_residuals(ts, pp_p, 'Poisson Process', country, output_dir, likelihood='poisson', lags=1, color='gray') 
        ks_ihp_stat, ks_ihp_pval, lb_ihp_stat, lb_ihp_pval = evaluate_discrete_residuals(ts, ihp_p, 'Inhomogeneous Poisson', country, output_dir, likelihood='poisson', lags=1, color='brown') 
        ks_ar_stat, ks_ar_pval, lb_ar_stat, lb_ar_pval = evaluate_discrete_residuals(ts, ar_p, 'Discrete AR(7)', country, output_dir, likelihood='poisson', lags=7, color='orange')
        ks_inar_stat, ks_inar_pval, lb_inar_stat, lb_inar_pval = evaluate_discrete_residuals(ts, inar_p, 'INAR(1)', country, output_dir, likelihood='poisson', lags=1, color='teal')
        ks_hp_stat, ks_hp_pval, lb_hp_stat, lb_hp_pval = evaluate_discrete_residuals(ts, hp_p, 'Hawkes (Pois)', country, output_dir, likelihood='poisson', lags=1, color='green')
        ks_hnb_stat, ks_hnb_pval, lb_hnb_stat, lb_hnb_pval = evaluate_discrete_residuals(ts, hnb_p, 'Hawkes (NB Exp)', country, output_dir, likelihood='neg_binomial', r_disp=hnb_params['r'], lags=1, color='red')
        ks_hpow_stat, ks_hpow_pval, lb_hpow_stat, lb_hpow_pval = evaluate_discrete_residuals(ts, hpow_p, 'Hawkes (NB Pow)', country, output_dir, likelihood='neg_binomial', r_disp=hpow_params['r'], lags=1, color='purple') 
        ks_hmm_stat, ks_hmm_pval, lb_hmm_stat, lb_hmm_pval = evaluate_discrete_residuals(ts, hmm_p, 'HMM (2-State)', country, output_dir, likelihood='poisson', lags=1, color='blue')

        # ==========================================
        # 3. Create SUPER Dictionary & Calculate Metrics
        # ==========================================
        models = {
            'Poisson Process': (pp_p, pp_l, pp_u, pp_params, pp_ll, pp_k, ks_pp_stat, ks_pp_pval, lb_pp_stat, lb_pp_pval),
            'Inhomogeneous Poisson': (ihp_p, ihp_l, ihp_u, ihp_params, ihp_ll, ihp_k, ks_ihp_stat, ks_ihp_pval, lb_ihp_stat, lb_ihp_pval),
            'Discrete AR(7)': (ar_p, ar_l, ar_u, ar_params, ar_ll, ar_k, ks_ar_stat, ks_ar_pval, lb_ar_stat, lb_ar_pval),
            'INAR(1)': (inar_p, inar_l, inar_u, inar_params, inar_ll, inar_k, ks_inar_stat, ks_inar_pval, lb_inar_stat, lb_inar_pval),
            'Hawkes (Pois Exp)': (hp_p, hp_l, hp_u, hp_params, hp_ll, hp_k, ks_hp_stat, ks_hp_pval, lb_hp_stat, lb_hp_pval),
            'Hawkes (NB Exp)': (hnb_p, hnb_l, hnb_u, hnb_params, hnb_ll, hnb_k, ks_hnb_stat, ks_hnb_pval, lb_hnb_stat, lb_hnb_pval),
            'Hawkes (NB Pow)': (hpow_p, hpow_l, hpow_u, hpow_params, hpow_ll, hpow_k, ks_hpow_stat, ks_hpow_pval, lb_hpow_stat, lb_hpow_pval),
            'HMM (2-State)': (hmm_p, hmm_l, hmm_u, hmm_params, hmm_ll, hmm_k, ks_hmm_stat, ks_hmm_pval, lb_hmm_stat, lb_hmm_pval)
        }
        
        results = []
        for name, (preds, lower, upper, params, ll, k, ks_stat, p_val, lb_stat, lb_pval) in models.items():
            aic, bic = calc_ic(ll, k, n)  # Assumes calc_ic exists in your code

            # Unpack the dict, cast to standard float, and join into a clean string
            clean_params = ", ".join([f"{key}: {float(val)}" for key, val in params.items()])

            results.append({
                'Model': name,
                'BIC': round(bic, 1),
                'MASE': round(calc_mase(ts, preds), 2),  
                'MDA': f"{calc_mda(ts, preds)*100:.1f}%", 
                'K-S Stat': round(ks_stat, 3),
                'K-S p-value': f"{p_val:.3e}",
                'Ljung-Box Stat': round(lb_stat, 3),
                'Ljung-Box p-value': f"{lb_pval:.3e}",
                'Parameters': clean_params
            })

        # Display the leaderboard! Because they all use discrete log-likelihoods, 
        # BIC is now a 100% fair mathematical comparison across all 8 models.
        print(pd.DataFrame(results).to_string(index=False))
        
        # ==========================================
        # 4. Plotting Generation
        # ==========================================
        # --- ORIGINAL COMBINED PLOT (No Uncertainty) ---
        plt.figure(figsize=(15, 6))
        plt.plot(full_dates, ts, label='News', color='black', alpha=0.4, linewidth=3)
        plt.plot(full_dates, pp_p, label='Poisson', color='gray', linestyle='--', alpha=0.8) 
        plt.plot(full_dates, ihp_p, label='Inhom. Poisson', color='brown', linestyle=':', alpha=0.8)  
        plt.plot(full_dates, ar_p, label='AR(7)', color='orange', linestyle='-', alpha=0.8)
        plt.plot(full_dates, inar_p, label='INAR(1)', color='teal', linestyle='-', alpha=0.8)
        plt.plot(full_dates, hp_p, label='Hawkes (Pois)', color='green', linestyle='-.', alpha=0.8)
        plt.plot(full_dates, hnb_p, label='Hawkes (NB)', color='red', linestyle='--', alpha=0.9)
        plt.plot(full_dates, hpow_p, label='Hawkes (NB-Pow)', color='purple', linestyle='-.', alpha=0.9) 
        plt.plot(full_dates, hmm_p, label='HMM (2)', color='blue', linestyle=':', alpha=0.9)
        
        plt.title(f'News Volume Modeling: {country} (2024)')
        plt.xlabel('Date')
        plt.ylabel('Daily Articles')
        plt.legend(loc='best') # Moved legend slightly outside to prevent overlap
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename_main = os.path.join(output_dir, f"{country}_main_fit_2024.png")
        plt.savefig(filename_main, dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- UNCERTAINTY QUANTIFICATION GRID PLOT ---
        fig, axes = plt.subplots(4, 2, figsize=(16, 15), sharex=True, sharey=True) 
        fig.suptitle(f'Uncertainty Quantification (95% CI): {country} (2024)', fontsize=16)
        axes = axes.flatten()
        
        # Ensure 8 colors for the 8 models
        colors = ['gray', 'brown', 'orange', 'teal', 'green', 'red', 'purple', 'blue'] 
        
        for i, (name, (preds, lower, upper, params, ll, k, ks_stat, p_val, lb_stat, lb_pval)) in enumerate(models.items()):
            ax = axes[i]
            # Plot Actuals
            ax.plot(full_dates, ts, color='black', alpha=0.3, label='Actual Data')
            # Plot Prediction Line
            ax.plot(full_dates, preds, color=colors[i], label=f'{name} Mean')
            # Plot Shaded Uncertainty Interval
            ax.fill_between(full_dates, lower, upper, color=colors[i], alpha=0.2, label='95% Confidence Interval')
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            # Formatting logic for a 4x2 grid
            if i >= 6: ax.set_xlabel('Date') # Only print 'Date' on the bottom 2 subplots
            if i % 2 == 0: ax.set_ylabel('Daily Articles') # Only print 'Y' on the left side
            
        plt.tight_layout()
        filename_uq = os.path.join(output_dir, f"{country}_uq_grid_2024.png")
        plt.savefig(filename_uq, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"--> Saved main plot as {filename_main}")
        print(f"--> Saved UQ grid plot as {filename_uq}")


        
        '''# Extract and print the actual Catalyst anomalies!
        hnb_preds, hnb_params = hnb_p, hnb_params
        catalysts_df = extract_hawkes_catalysts(full_dates, ts, hnb_preds, hnb_params['mu'], top_n=5)
        print(f"\n--- TOP 5 PATIENT ZERO CATALYSTS (Mainshocks) FOR {country} ---")
        # Format ratio for console output only
        display_df = catalysts_df.copy()
        display_df['Organic_Ratio'] = display_df['Organic_Ratio'].map('{:,.1f}%'.format)
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        print(display_df.to_string(index=False))

        # 7. Plot 2: Catalyst Specific Plot (Requested Visualization)
        # Focus plot: Raw Data + HNB Intensity + Catalyst Vertical Lines
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(full_dates, ts, label='Actual News Counts', color='black', alpha=0.4, linewidth=2)
        ax.plot(full_dates, hnb_preds, label='Hawkes (NB Exp) Intensity', color='red', linestyle='--', linewidth=2)
        
        # Draw vertical lines for each catalyst
        # Add labels to the lines to make them clearer
        first_line = True
        for idx, row in catalysts_df.iterrows():
            ax.axvline(x=row['Date'], color='red', linestyle=':', alpha=0.7, linewidth=1.5, 
                       label='Catalyst Event (Mainshock)' if first_line else "")
            first_line = False
            
        ax.set_title(f'Catalyst Identification (Stochastic Declustering): {country} (2024)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11); ax.set_ylabel('Daily Articles', fontsize=11)
        ax.legend(loc='upper right', fontsize=10); ax.grid(alpha=0.2)
        plt.tight_layout()
        filename_cat = os.path.join(output_dir, f"{country}_catalysts_2024.png")
        plt.savefig(filename_cat, dpi=300); plt.close()

        print(f"\n--> Saved Catalyst plot to {filename_cat}")
        print(f"--> Saved UQ grid plot to {filename_uq}")'''