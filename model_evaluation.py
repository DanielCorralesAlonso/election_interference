import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from hmmlearn.hmm import PoissonHMM
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom, norm, kstest, probplot
import warnings
import os

warnings.filterwarnings("ignore")

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
def fit_ar(ts, lags=7):
    log_ts = np.log1p(ts)
    model = AutoReg(log_ts, lags=lags).fit()
    
    # Get predictions and Confidence Intervals in log space
    pred_obj = model.get_prediction(start=lags, end=len(ts)-1, dynamic=False)
    log_preds = pred_obj.predicted_mean
    log_ci = pred_obj.conf_int(alpha=0.05) # 95% CI
    
    # FIX: Ensure log_ci is treated as a numpy array so slicing [:, 0] works universally
    if hasattr(log_ci, 'values'):
        log_ci = log_ci.values
        
    # Reconstruct arrays to match original length
    preds = np.zeros(len(ts))
    lower = np.zeros(len(ts))
    upper = np.zeros(len(ts))
    
    # Fill first 'lags' days with actuals (no uncertainty)
    preds[:lags] = lower[:lags] = upper[:lags] = ts[:lags]
    
    # FIX: Exponentiate using standard numpy array indexing [:, 0] instead of .iloc
    preds[lags:] = np.maximum(np.expm1(log_preds), 0)
    lower[lags:] = np.maximum(np.expm1(log_ci[:, 0]), 0) 
    upper[lags:] = np.maximum(np.expm1(log_ci[:, 1]), 0) 
    
    # FIX: Make parameter extraction robust to both Pandas Series and Numpy Arrays
    p_array = model.params.values if hasattr(model.params, 'values') else model.params
    params = {'intercept': round(p_array[0], 2), 'lag_1': round(p_array[1], 2)}
    
    return preds, lower, upper, params, model.llf, len(model.params)

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




def evaluate_ar_residuals(actual, predicted, lags, model_name, country_name, output_dir):
    y_act_log = np.log1p(actual[lags:])
    y_pred_log = np.log1p(predicted[lags:])
    raw_res = y_act_log - y_pred_log
    std_res = (raw_res - np.mean(raw_res)) / (np.std(raw_res) + 1e-10)
    
    ks_stat, p_val = kstest(std_res, 'norm')
    _plot_qq(std_res, model_name, country_name, ks_stat, p_val, 'orange', output_dir)
    return ks_stat, p_val

def evaluate_poisson_residuals(actual, predicted_means, model_name, country_name, output_dir):
    n = len(actual)
    residuals = np.zeros(n)
    np.random.seed(42)
    for t in range(n):
        y = actual[t]
        lam = max(predicted_means[t], 1e-5)
        cdf_y = poisson.cdf(y, lam)
        cdf_y_minus_1 = 0.0 if y == 0 else poisson.cdf(y - 1, lam)
        u = np.clip(np.random.uniform(cdf_y_minus_1, cdf_y), 1e-10, 1 - 1e-10)
        residuals[t] = norm.ppf(u)
        
    ks_stat, p_val = kstest(residuals, 'norm')
    _plot_qq(residuals, model_name, country_name, ks_stat, p_val, 'blue', output_dir)
    return ks_stat, p_val

def evaluate_nbinom_residuals(actual, predicted_means, r_disp, model_name, country_name, output_dir):
    n = len(actual)
    residuals = np.zeros(n)
    np.random.seed(42)
    for t in range(n):
        y = actual[t]
        mu = max(predicted_means[t], 1e-5)
        p = r_disp / (r_disp + mu)
        cdf_y = nbinom.cdf(y, r_disp, p)
        cdf_y_minus_1 = 0.0 if y == 0 else nbinom.cdf(y - 1, r_disp, p)
        u = np.clip(np.random.uniform(cdf_y_minus_1, cdf_y), 1e-10, 1 - 1e-10)
        residuals[t] = norm.ppf(u)
        
    ks_stat, p_val = kstest(residuals, 'norm')
    _plot_qq(residuals, model_name, country_name, ks_stat, p_val, 'red', output_dir)
    return ks_stat, p_val

def _plot_qq(residuals, model_name, country_name, ks_stat, p_val, color, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linestyle('--')
    
    ax.set_title(f'{model_name} Q-Q: {country_name}\nK-S: {ks_stat:.3f} | p-val: {p_val:.3e}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{country_name}_{model_name.replace(' ', '_')}_QQ.png"), dpi=150)
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
            
        # 1. Fit models 
        ar_p, ar_l, ar_u, ar_params, ar_ll, ar_k = fit_ar(ts, lags=7)
        pp_p, pp_l, pp_u, pp_params, pp_ll, pp_k = fit_poisson_process(ts) # NEW
        ihp_p, ihp_l, ihp_u, ihp_params, ihp_ll, ihp_k = fit_inhomogeneous_poisson(ts) # NEW
        hp_p, hp_l, hp_u, hp_params, hp_ll, hp_k = fit_hawkes(ts, decay=0.8, likelihood='poisson')
        hnb_p, hnb_l, hnb_u, hnb_params, hnb_ll, hnb_k = fit_hawkes(ts, decay=0.8, likelihood='neg_binomial')
        hpow_p, hpow_l, hpow_u, hpow_params, hpow_ll, hpow_k = fit_hawkes_powerlaw(ts) # NEW
        hmm_p, hmm_l, hmm_u, hmm_params, hmm_ll, hmm_k = fit_hmm(ts, n_components=2)
        
        # 2. Evaluate Residuals & K-S Tests
        ks_ar_stat, ks_ar_pval = evaluate_ar_residuals(ts, ar_p, 7, 'AR(7)', country, output_dir)
        ks_pp_stat, ks_pp_pval = evaluate_poisson_residuals(ts, pp_p, 'Poisson Process', country, output_dir) # NEW
        ks_ihp_stat, ks_ihp_pval = evaluate_poisson_residuals(ts, ihp_p, 'Inhomogeneous Poisson', country, output_dir) # NEW
        ks_hp_stat, ks_hp_pval = evaluate_poisson_residuals(ts, hp_p, 'Hawkes (Pois)', country, output_dir)
        ks_hnb_stat, ks_hnb_pval = evaluate_nbinom_residuals(ts, hnb_p, hnb_params['r'], 'Hawkes (NB Exp)', country, output_dir)
        ks_hpow_stat, ks_hpow_pval = evaluate_nbinom_residuals(ts, hpow_p, hpow_params['r'], 'Hawkes (NB Pow)', country, output_dir) # NEW
        ks_hmm_stat, ks_hmm_pval = evaluate_poisson_residuals(ts, hmm_p, 'HMM (2-State)', country, output_dir)

        # 3. Create SUPER Dictionary packing all variables
        models = {
            'AR(7) *Log*': (ar_p, ar_l, ar_u, ar_params, ar_ll, ar_k, ks_ar_stat, ks_ar_pval),
            'Poisson Process': (pp_p, pp_l, pp_u, pp_params, pp_ll, pp_k, ks_pp_stat, ks_pp_pval),
            'Inhomogeneous Poisson': (ihp_p, ihp_l, ihp_u, ihp_params, ihp_ll, ihp_k, ks_ihp_stat, ks_ihp_pval),
            'Hawkes (Pois Exp)': (hp_p, hp_l, hp_u, hp_params, hp_ll, hp_k, ks_hp_stat, ks_hp_pval),
            'Hawkes (NB Exp)': (hnb_p, hnb_l, hnb_u, hnb_params, hnb_ll, hnb_k, ks_hnb_stat, ks_hnb_pval),
            'Hawkes (NB Pow)': (hpow_p, hpow_l, hpow_u, hpow_params, hpow_ll, hpow_k, ks_hpow_stat, ks_hpow_pval),
            'HMM (2-State)': (hmm_p, hmm_l, hmm_u, hmm_params, hmm_ll, hmm_k, ks_hmm_stat, ks_hmm_pval)
        }
        # 3. Calculate metrics and build DataFrame
        results = []
        for name, (preds, lower, upper, params, ll, k, ks_stat, p_val) in models.items():
            aic, bic = calc_ic(ll, k, n)
            results.append({
                'Model': name,
                'BIC': round(bic, 1),
                'MASE': round(calc_mase(ts, preds), 2),
                'MDA': f"{calc_mda(ts, preds)*100:.1f}%",
                'K-S Stat': round(ks_stat, 3),
                'K-S p-value': f"{p_val:.3e}",
                'Parameters': str(params)
            })

        
            
        print(pd.DataFrame(results).to_string(index=False))
        
        # --- 1. ORIGINAL COMBINED PLOT (No Uncertainty) ---
        plt.figure(figsize=(15, 6))
        plt.plot(full_dates, ts, label='Actual News', color='black', alpha=0.4, linewidth=3)
        plt.plot(full_dates, ar_p, label='AR(7)', color='orange', linestyle='-', alpha=0.8)
        plt.plot(full_dates, pp_p, label='Poisson Process', color='gray', linestyle='--', alpha=0.8) # NEW
        plt.plot(full_dates, ihp_p, label='Inhomogeneous Poisson', color='brown', linestyle=':', alpha=0.8) # NEW   
        plt.plot(full_dates, hp_p, label='Hawkes (Poisson)', color='green', linestyle='-.', alpha=0.8)
        plt.plot(full_dates, hnb_p, label='Hawkes (Negative Binomial)', color='red', linestyle='--', alpha=0.9)
        plt.plot(full_dates, hpow_p, label='Hawkes (NB Power Law)', color='purple', linestyle='-.', alpha=0.9) # NEW
        plt.plot(full_dates, hmm_p, label='HMM (2-State)', color='blue', linestyle=':', alpha=0.9)
        
        plt.title(f'News Volume Modeling: {country} (2024)')
        plt.xlabel('Date')
        plt.ylabel('Daily Articles')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename_main = os.path.join(output_dir, f"{country}_main_fit_2024.png")
        plt.savefig(filename_main, dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- 2. NEW UNCERTAINTY QUANTIFICATION GRID PLOT ---
        fig, axes = plt.subplots(4, 2, figsize=(16, 15), sharex=True, sharey=True) # Changed to 4x2 and increased height
        fig.suptitle(f'Uncertainty Quantification (95% CI): {country} (2024)', fontsize=16)
        axes = axes.flatten()
        
        colors = ['orange', 'gray', 'brown', 'green', 'red', 'purple', 'blue'] # Added colors
        
        for i, (name, (preds, lower, upper, params, ll, k, ks_stat, p_val)) in enumerate(models.items()):
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
            if i >= 2: ax.set_xlabel('Date')
            if i % 2 == 0: ax.set_ylabel('Daily Articles')
            
        plt.tight_layout()
        filename_uq = os.path.join(output_dir, f"{country}_uq_grid_2024.png")
        plt.savefig(filename_uq, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"--> Saved main plot as {filename_main}")
        print(f"--> Saved UQ grid plot as {filename_uq}")


        
        # Extract and print the actual Catalyst anomalies!
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
        print(f"--> Saved UQ grid plot to {filename_uq}")