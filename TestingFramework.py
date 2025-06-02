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
    Binomial test for marginal coverage of prediction intervals.

    Tests whether the observed violation count significantly deviates
    from the expected count under Binomial(n, alpha), where violations
    are i.i.d. Bernoulli(alpha).

    Parameters
    ----------
    violation_series : array-like of {0, 1}
        Binary vector; 1 if true value falls outside interval, else 0.
    alpha : float
        Nominal miscoverage rate (e.g., 0.1 for 90% coverage).

    Returns
    -------
    p_under : float
        P-value for H0: coverage ≤ (1 - alpha) — tests undercoverage.
    p_over : float
        P-value for H0: coverage ≥ (1 - alpha) — tests overconservatism.
    p_two_sided : float
        Nonparametric p-value for any deviation from expected coverage.
    """
    n = len(violation_series)          # Total predictions
    H_obs = np.sum(violation_series)   # Count of violations

    # PMF of observing exactly H_obs violations under H0
    prob_H_obs = stats.binom.pmf(H_obs, n, alpha)

    # Two-sided: sum of all probs as or less likely than observed
    probs = stats.binom.pmf(np.arange(n + 1), n, alpha)
    p_two_sided = np.sum(probs[probs <= prob_H_obs])

    # Right tail: too many violations (undercoverage)
    p_under = stats.binom.sf(H_obs - 1, n, alpha)  # P(X ≥ H_obs)

    # Left tail: too few violations (overconservative)
    p_over = stats.binom.cdf(H_obs, n, alpha)      # P(X ≤ H_obs)

    return p_under, p_over, p_two_sided

# ---------------------------
# 2. Geometric-Conformal (Weibull Run-length Tests)
# ---------------------------
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

def geometric_conformal_test(violation_series, alpha, eps: float = 1e-12):
    """
    Discrete-Weibull run-length back-test for prediction-interval validity.

    Context
    -------
    If interval forecasts are calibrated, the hit sequence Iₜ is i.i.d. Bernoulli(α).
    The inter-hit durations Dᵢ should follow a **geometric** distribution with
    constant hazard α.  We nest this null inside a two-parameter Weibull hazard

        k(d | a, b) = a·d^{b-1}, 0 < a < 1, b > 0,

    and test:
        • UC   (a ≠ α, b = 1)
        • IND  (b ≠ 1, a free)
        • JOINT(a ≠ α ∨ b ≠ 1)

    Censoring: the first and last durations are right-censored whenever the sample
    does not start / end with a violation.

    Parameters
    ----------
    violation_series : 1-D array-like of {0,1}
        1 → interval violated at t, 0 → covered.
    alpha : float
        Nominal mis-coverage rate (e.g. 0.05 for 95 % PIs).
    eps : float, default 1e-12
        Floor to keep log / prod numerically safe.

    Returns
    -------
    pval_LRUC   : float   # unconditional-coverage p-value   (df = 1)
    pval_LRInd  : float   # duration-independence p-value    (df = 1)
    pval_LRJoint: float   # joint test p-value              (df = 2)
    """
    # ─── Basic checks ──────────────────────────────────────────────────────────
    failures = np.where(violation_series == 1)[0]
    if len(failures) < 2:
        raise ValueError("≥2 violations required for run-length analysis.")

    run_lengths = np.diff(failures)
    if np.any(run_lengths < 1):
        raise ValueError("Run-length < 1 found — duplicated violation index?")

    N = len(run_lengths)

    # Censoring flags: right-censor first/last duration if series starts/ends in-coverage
    first_censored = failures[0] > 0
    last_censored  = failures[-1] < len(violation_series) - 1
    C = np.zeros(N, dtype=int)
    C[0], C[-1] = first_censored, last_censored

    # ─── Hazard / survival helpers ─────────────────────────────────────────────
    def hazard(d, a, b):
        return np.clip(a * d ** (b - 1), eps, 1 - eps)

    def survival(d, a, b):
        if d <= 1:
            return 1.0
        τ = np.arange(1, d)
        return np.clip(np.prod(1 - hazard(τ, a, b)), eps, 1)

    # ─── Log-likelihood (matches Pelletier & Wei 2016, Sec. 2) ───────────────
    def loglike(a, b):
        logL = 0.0

        # first duration
        d = run_lengths[0]
        S = survival(d, a, b)
        if C[0]:
            logL += np.log(S)
        else:
            logL += np.log(hazard(d, a, b)) + np.log(S)

        # middle durations (fully observed)
        for d in run_lengths[1:-1]:
            k = hazard(d, a, b)
            S = survival(d, a, b)
            logL += np.log(k) + np.log(S)

        # last duration
        d = run_lengths[-1]
        S = survival(d, a, b)
        if C[-1]:
            logL += np.log(S)
        else:
            logL += np.log(hazard(d, a, b)) + np.log(S)

        return logL

    # ─── Log-likelihoods under null / alternatives ────────────────────────────
    ll_null  = loglike(alpha, 1.0)                              
    ll_uc    = -minimize(lambda a: -loglike(a[0], 1.0),
                         x0=[alpha], bounds=[(eps, 1 - eps)]).fun
    ll_joint = -minimize(lambda p: -loglike(p[0], p[1]),
                         x0=[alpha, 1.0],
                         bounds=[(eps, 1 - eps), (eps, 10)]).fun

    # LR statistics
    LR_UC    = -2 * (ll_null - ll_uc)
    LR_Ind   = -2 * (ll_uc   - ll_joint)
    LR_Joint = -2 * (ll_null - ll_joint)

    # p-values, χ² reference
    p_uc    = stats.chi2.sf(LR_UC,    df=1)
    p_ind   = stats.chi2.sf(LR_Ind,   df=1)
    p_joint = stats.chi2.sf(LR_Joint, df=2)

    return p_uc, p_ind, p_joint

# ---------------------------
# 3. Dynamic Binary Logistic Regression Test
# ---------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def dynamic_binary_regression_test(
    y_true,
    violation_series,
    lower_bounds,
    upper_bounds,
    alpha,
    p: int = 4,
    q: int = 4,
    r: int = 4,
    s: int = 4,
    explained_variance_threshold: float = 0.95,
):
    """
    Dynamic Binary Logistic Regression Backtest for Prediction Interval Validity.

    This test evaluates whether violations of a conformal predictor are:
    (1) serially independent (Bernoulli-distributed), and
    (2) conditionally i.i.d. Bernoulli(α) given past model state.

    A logistic model is fit to the violation series using lagged features.
    PCA is applied for dimensionality reduction before statistical testing.

    Hypotheses
    ----------
    H0_ind  : Violations are i.i.d. Bernoulli(π)
    H0_cc   : Violations are i.i.d. Bernoulli(α) even conditional on past features

    Parameters
    ----------
    y_true : array-like
        Observed target values yₜ over time.
    violation_series : array-like of {0, 1}
        Indicator vector where 1 = violation (yₜ outside PI), 0 = within bounds.
    lower_bounds, upper_bounds : array-like
        Lower and upper prediction interval bounds at each time t.
    alpha : float
        Nominal miscoverage rate (e.g., 0.05 for 95% intervals).
    p : int
        Number of lags for past violations Iₜ.
    q : int
        Number of lags for bounds and their interactions with Iₜ.
    r : int
        Number of lags of predicted risk π̂ₜ to include as features.
    s : int
        Number of lags for y_true and y_true × Iₜ interactions.
    explained_variance_threshold : float
        Minimum variance to retain in PCA (between 0 and 1).

    Returns
    -------
    p_ind : float
        p-value for the independence test (H0: Bernoulli π).
    p_cc : float
        p-value for the conditional coverage test (H0: Bernoulli α).
    """
    n = len(violation_series)
    max_lag = max(p, q, r, s)

    if n <= max_lag + 1 or np.sum(violation_series) < 10:
        return np.nan, np.nan  # not enough data to run the test reliably

    # Step 1: Construct lagged feature matrix (X)
    base_X = []
    for t in range(max_lag, n):
        f = list(violation_series[t - np.arange(1, p + 1)])
        for i in range(1, q + 1):
            I = violation_series[t - i]
            lb, ub = lower_bounds[t - i], upper_bounds[t - i]
            f.extend([lb, ub, lb * I, ub * I])
        for i in range(1, s + 1):
            yt, I = y_true[t - i], violation_series[t - i]
            f.extend([yt, yt * I])
        base_X.append(f)

    base_X = np.asarray(base_X)
    y = violation_series[max_lag:]

    # Step 2: Fit first-stage logit model to estimate π̂ₜ
    clf_stage1 = LogisticRegression(penalty = None, solver="lbfgs", max_iter=300)
    clf_stage1.fit(base_X, y)
    pi_hat = clf_stage1.predict_proba(base_X)[:, 1]

    # Step 3: Add r lags of predicted π̂ₜ to feature matrix
    pi_hat_full = np.concatenate([np.full(max_lag, pi_hat.mean()), pi_hat])
    X = []
    for t in range(max_lag, n):
        f = list(base_X[t - max_lag])
        f.extend(pi_hat_full[t - np.arange(1, r + 1)])
        X.append(f)
    X = np.asarray(X)

    # Step 4: Standardize features (z-score normalization)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    # Step 5: Dimensionality reduction via PCA
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cumvar, explained_variance_threshold) + 1
    X_red = pca.transform(X)[:, :k]

    # Step 6: Fit full dynamic logistic model
    clf_full = LogisticRegression(penalty = None, solver="lbfgs", max_iter=300)
    clf_full.fit(X_red, y)
    p_full = clf_full.predict_proba(X_red)[:, 1]
    eps = 1e-10
    ll_full = np.sum(y * np.log(p_full + eps) + (1 - y) * np.log(1 - p_full + eps))

    # Step 7a: Null model — intercept-only (tests independence)
    clf_null = LogisticRegression(penalty = None, solver="lbfgs", fit_intercept=True, max_iter=300)
    clf_null.fit(np.zeros((len(y), 1)), y)
    p_indep = clf_null.predict_proba(np.zeros((len(y), 1)))[:, 1]
    ll_ind = np.sum(y * np.log(p_indep + eps) + (1 - y) * np.log(1 - p_indep + eps))
    LR_ind = -2 * (ll_ind - ll_full)
    p_ind = stats.chi2.sf(LR_ind, df=k) # degrees of freedom = number of features

    # Step 7b: Null model — fixed α (tests conditional coverage)
    ll_alpha = np.sum(y * np.log(alpha) + (1 - y) * np.log(1 - alpha))
    LR_cc = -2 * (ll_alpha - ll_full)
    p_cc = stats.chi2.sf(LR_cc, df=k + 1) # degrees of freedom = number of features + intercept

    return p_ind, p_cc



# ---------------------------
# Diebold-Mariano Interval-Score Test
# ---------------------------
def diebold_mariano_test(
    lower_bounds1, upper_bounds1,
    lower_bounds2, upper_bounds2,
    y_true, alpha,
    verbose: bool = True
):
    """
    Two-model Diebold–Mariano test using the Interval Score (Gneiting & Raftery, 2007).

    H0 (two-sided)
    --------------
    E[S₁ − S₂] = 0   # equal predictive quality of the two interval forecasts

    Implementation notes
    --------------------
    • Interval Score penalises both width and violations; lower is better.  
    • HAC variance estimated with Bartlett weights and lag ⌊T^{1/3}⌋ (Andrews, 1991).  
    • Test statistic → 𝒩(0,1) under H0.

    Parameters
    ----------
    lower_bounds1, upper_bounds1 : array-like
        Prediction-interval bounds from model 1.
    lower_bounds2, upper_bounds2 : array-like
        Prediction-interval bounds from model 2.
    y_true : array-like
        Realised values yₜ.
    alpha : float
        Nominal miscoverage rate (e.g. 0.05 for 95 % intervals).
    verbose : bool, default True
        If True, prints d̄, HAC variance, statistic, p-value.

    Returns
    -------
    DM_stat : float
        Standardised Diebold–Mariano statistic.
    p_value : float
        Two-sided p-value (asymptotic 𝒩(0,1)).
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("y_true is empty.")
    alpha = np.clip(alpha, 1e-10, 1 - 1e-10)  # numerical safety

    # Lag truncation for HAC
    h_lag = int(np.floor(n ** (1 / 3)))

    # Interval Scores (lower is better)
    S1 = (upper_bounds1 - lower_bounds1) 
    + (2 / alpha) * (y_true - upper_bounds1) * (y_true > upper_bounds1) 
    + (2 / alpha) * (lower_bounds1 - y_true) * (y_true < lower_bounds1)

    S2 = (upper_bounds2 - lower_bounds2)  
    + (2 / alpha) * (y_true - upper_bounds2) * (y_true > upper_bounds2) 
    + (2 / alpha) * (lower_bounds2 - y_true) * (y_true < lower_bounds2)

    # Loss differential
    d = S1 - S2
    d_bar = d.mean()

    # Newey–West HAC variance with Bartlett kernel
    gamma0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for lag in range(1, h_lag + 1):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        weight = 1 - lag / (h_lag + 1)
        gamma_sum += 2 * weight * cov

    DM_var = (gamma0 + gamma_sum) / n
    if DM_var == 0:
        raise ValueError("HAC variance is zero; DM statistic undefined.")

    DM_stat = d_bar / np.sqrt(DM_var)
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))

    if verbose:
        print(f"Mean loss differential d̄   : {d_bar:.6f}")
        print(f"HAC variance (Newey-West)  : {DM_var:.6f}")
        print(f"Diebold-Mariano statistic  : {DM_stat:.3f}")
        print(f"p-value                    : {p_value:.4f}")

    return DM_stat, p_value