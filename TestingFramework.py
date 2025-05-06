import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# ---------------------------
# 1. Binomial Tests
# ---------------------------

def binomial_test(violation_series, alpha):
    """
    Perform a binomial test to evaluate if the number of observed violations
    significantly deviates from the expected number under the null hypothesis.

    Parameters:
    ----------
    violation_series : array-like
        Boolean or binary array indicating violations (1 = violation, 0 = no violation).
    alpha : float
        Expected violation probability under the null hypothesis (e.g. 0.05 for 5% VaR).

    Returns:
    -------
    p_under : float
        One-sided p-value for observing H_obs or more violations (right tail).
    p_over : float
        One-sided p-value for observing H_obs or fewer violations (left tail).
    p_two_sided : float
        Two-sided p-value based on probabilities less than or equal to the observed likelihood.
    """
    n = len(violation_series)                 # Total number of forecasts
    H_obs = np.sum(violation_series)          # Observed number of violations

    # Probability of observing exactly H_obs violations under Binomial(n, alpha)
    prob_H_obs = stats.binom.pmf(H_obs, n, alpha)

    # Two-sided p-value: sum of all probabilities ≤ prob_H_obs (nonparametric approach)
    probs = stats.binom.pmf(np.arange(n + 1), n, alpha)
    p_two_sided = np.sum(probs[probs <= prob_H_obs])

    # One-sided p-values
    p_under = stats.binom.sf(H_obs - 1, n, alpha)  # P(X ≥ H_obs)
    p_over = stats.binom.cdf(H_obs, n, alpha)      # P(X ≤ H_obs)

    return p_under, p_over, p_two_sided

# ---------------------------
# 2. Geometric-Conformal (Weibull Run-length Tests)
# ---------------------------
def geometric_conformal_test(violation_series, alpha):
    """
    Perform a geometric test for Conformal Prediction violations using run-length modeling.
    This test is analogous to Christoffersen's UC, IND, and Joint tests.

    Parameters
    ----------
    violation_series : array-like
        Binary array indicating violations (1 = violation, 0 = no violation).
    alpha : float
        Nominal error rate of the conformal predictor (e.g., 0.05 for 95% coverage).

    Returns
    -------
    pval_LRUC : float
        p-value for Unconditional Coverage (UC) test.
    pval_LRInd : float
        p-value for Independence (IND) test.
    pval_LRJoint : float
        p-value for Joint (UC + IND) test.
    """
    failures = np.where(violation_series == 1)[0]
    if len(failures) < 2:
        raise ValueError("Not enough violations to compute run-lengths.")

    # Run-lengths between consecutive violations, clipped to avoid zero intervals
    run_lengths = np.clip(np.diff(failures), 1, None)

    # Log-likelihood under generalized geometric model with shape (b) and rate (a)
    def neg_log_likelihood_ab(params):
        a, b = params
        if a <= 0 or a >= 1 or b <= 0:
            return np.inf
        ll = np.sum(np.log(a) + np.log(b) + (b - 1) * np.log(run_lengths) - a * run_lengths**b)
        return -ll

    # Log-likelihood under standard geometric model (b = 1)
    def neg_log_likelihood_a_only(a):
        if a <= 0 or a >= 1:
            return np.inf
        ll = np.sum(np.log(a) + (run_lengths - 1) * np.log(1 - a))
        return -ll

    # Null hypothesis: fixed a = alpha, b = 1
    ll_null = np.sum(np.log(alpha) + (run_lengths - 1) * np.log(1 - alpha))

    # UC test: estimate a freely, fix b = 1
    res_uc = minimize(neg_log_likelihood_a_only, x0=[alpha], bounds=[(1e-5, 1 - 1e-5)])
    a_hat_uc = res_uc.x[0]
    ll_uc = -neg_log_likelihood_a_only(a_hat_uc)

    # Full alternative: estimate both a and b
    res_full = minimize(neg_log_likelihood_ab, x0=[alpha, 1.0], bounds=[(1e-5, 1 - 1e-5), (1e-5, None)])
    a_hat_full, b_hat_full = res_full.x
    ll_full = -neg_log_likelihood_ab([a_hat_full, b_hat_full])

    # Likelihood ratio statistics
    LR_UC = -2 * (ll_null - ll_uc)
    LR_Ind = -2 * (ll_uc - ll_full)
    LR_Joint = -2 * (ll_null - ll_full)

    # Corresponding p-values
    pval_LRUC = 1 - stats.chi2.cdf(LR_UC, df=1)
    pval_LRInd = 1 - stats.chi2.cdf(LR_Ind, df=1)
    pval_LRJoint = 1 - stats.chi2.cdf(LR_Joint, df=2)

    return pval_LRUC, pval_LRInd, pval_LRJoint



# ---------------------------
# 3. Dynamic Binary Logistic Regression Test
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dynamic_binary_regression_test(y_true, violation_series, lower_bounds, upper_bounds, alpha, 
                                   p=4, q=4, r=4, s=4, explained_variance_threshold=0.95):
    """
    Dynamic Binary Regression Test with PCA for Conformal Prediction diagnostics.
    
    This test detects whether violations are dynamically predictable using logistic regression
    with lagged features and dimensionality reduction.

    Parameters
    ----------
    y_true : array-like
        True observed values of the target variable.
    violation_series : array-like
        Binary array indicating prediction interval violations.
    lower_bounds : array-like
        Lower bounds of prediction intervals.
    upper_bounds : array-like
        Upper bounds of prediction intervals.
    alpha : float
        Nominal error rate (e.g. 0.05 for 95% coverage).
    p, q, r, s : int
        Number of lags for violation indicators (p), bounds & interactions (q),
        predicted probabilities (r), and y_true & interactions (s).
    explained_variance_threshold : float
        Proportion of variance to retain in PCA (e.g. 0.95 for 95%).

    Returns
    -------
    p_value : float
        p-value from likelihood ratio test assessing dynamic predictability.
    """
    n = len(violation_series)
    min_lag = max(p, q, r, s)

    if np.sum(violation_series) < 10 or n <= min_lag + 1:
        return np.nan, None

    # Step 1: Construct base features
    base_X = []
    for t in range(min_lag, n):
        features = []

        # Lagged violation indicators
        for i in range(1, p + 1):
            features.append(violation_series[t - i])

        # Lagged bounds and interactions
        for i in range(1, q + 1):
            lb = lower_bounds[t - i]
            ub = upper_bounds[t - i]
            It = violation_series[t - i]
            features.extend([lb, ub, lb * It, ub * It])

        # Lagged true values and interactions with violation indicator
        for i in range(1, s + 1):
            yt_i = y_true[t - i]
            It = violation_series[t - i]
            features.extend([yt_i, It * yt_i])

        base_X.append(features)

    base_X = np.array(base_X)
    y = violation_series[min_lag:]

    # Step 2: First-stage logistic regression (estimates predicted violation probabilities)
    clf_stage1 = LogisticRegression(solver='lbfgs', max_iter=300)
    clf_stage1.fit(base_X, y)
    pi_hat = clf_stage1.predict_proba(base_X)[:, 1]

    # Pad the beginning of the series with the mean probability for lag use
    pi_hat_full = np.concatenate([np.full(min_lag, np.mean(y)), pi_hat])

    # Step 3: Add lagged predicted probabilities as additional features
    X = []
    for t in range(min_lag, n):
        features = list(base_X[t - min_lag])
        for i in range(1, r + 1):
            features.append(pi_hat_full[t - i])
        X.append(features)

    X = np.array(X)

    # Step 4: Standardize the features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X = (X - X_mean) / X_std

    # Step 5: Apply PCA for dimensionality reduction
    pca = PCA()
    X_pca = pca.fit_transform(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cumulative_variance, explained_variance_threshold) + 1
    X_reduced = X_pca[:, :k]

    # Step 6: Second-stage logistic regression on PCA-reduced features
    clf_final = LogisticRegression(solver='lbfgs', max_iter=300)
    clf_final.fit(X_reduced, y)
    probs = clf_final.predict_proba(X_reduced)[:, 1]

    # Step 7: Compute log-likelihoods of full and restricted models
    eps = 1e-8  # for numerical stability
    ll_full = np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    ll_restricted = np.sum(y * np.log(alpha) + (1 - y) * np.log(1 - alpha))

    # Step 8: Likelihood ratio test
    LR_stat = -2 * (ll_restricted - ll_full)
    p_value = 1 - stats.chi2.cdf(LR_stat, df=k + 1)  # +1 for intercept

    return p_value





# ---------------------------
# 4. Diebold-Mariano Interval Score Test
# ---------------------------
from scipy import stats
import numpy as np

def diebold_mariano_test(lower_bounds1, upper_bounds1, lower_bounds2, upper_bounds2, y_true, alpha, verbose=True):
    """
    Diebold-Mariano Test for comparing the predictive efficiency of two interval forecasting methods
    using the Interval Score as a strictly proper scoring rule.

    Parameters
    ----------
    lower_bounds1, upper_bounds1 : array-like
        Lower and upper prediction bounds from model 1.
    lower_bounds2, upper_bounds2 : array-like
        Lower and upper prediction bounds from model 2.
    y_true : array-like
        True observed values.
    alpha : float
        Nominal miscoverage rate (e.g., 0.05 for 95% intervals).
    verbose : bool, optional
        If True, prints detailed diagnostics. Default is True.

    Returns
    -------
    DM_stat : float
        Diebold-Mariano test statistic.
    p_value : float
        Two-sided p-value under the standard normal distribution.
    """
    n = len(y_true)
    
    # Lag truncation h following Andrews (1991): floor(T^{1/3}) for HAC estimation
    h_lag = int(np.floor(n ** (1/3)))

    # Interval Scores (Gneiting & Raftery, 2007)
    S1 = (upper_bounds1 - lower_bounds1) \
        + (2 / alpha) * (y_true - upper_bounds1) * (y_true > upper_bounds1) \
        + (2 / alpha) * (lower_bounds1 - y_true) * (y_true < lower_bounds1)

    S2 = (upper_bounds2 - lower_bounds2) \
        + (2 / alpha) * (y_true - upper_bounds2) * (y_true > upper_bounds2) \
        + (2 / alpha) * (lower_bounds2 - y_true) * (y_true < lower_bounds2)

    # Loss differential
    d = S1 - S2
    d_bar = np.mean(d)

    # Newey-West HAC variance estimator (Bartlett weights)
    gamma0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0
    for lag in range(1, h_lag + 1):
        gamma_k = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        weight = 1 - lag / (h_lag + 1)
        gamma_sum += 2 * weight * gamma_k

    DM_var = (gamma0 + gamma_sum) / n
    DM_stat = d_bar / np.sqrt(DM_var)
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))  # two-sided test

    if verbose:
        print(f"Mean of d: {d_bar}")
        print(f"DM variance: {DM_var}")
        print(f"DM stat: {DM_stat}, p-value: {p_value}")

    return DM_stat, p_value
