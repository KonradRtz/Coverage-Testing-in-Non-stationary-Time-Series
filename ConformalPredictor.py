import numpy as np
import pandas as pd
from collections import deque

class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def _make_final_dataframe(self, lower_bounds, upper_bounds, y_true, violations):
        df = pd.DataFrame({
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'y_true': y_true,
            'violation': violations,
        })

        # Use pandas .shift() for lags
        for lag in range(1, 5):
            df[f'lower_bound_t-{lag}'] = df['lower_bound'].shift(lag)
            df[f'upper_bound_t-{lag}'] = df['upper_bound'].shift(lag)
            df[f'violation_t-{lag}'] = df['violation'].shift(lag)

        # Drop the rows with missing values caused by shifting
        df = df.dropna().reset_index(drop=True)

        return df

    # --------------------------------
    # 1. Split Conformal Prediction 
    # --------------------------------
    def standard_cp(self, pred_calib, y_calib, pred_test, y_test):
        """
        Compute split conformal prediction intervals using absolute residuals 
        from a held-out calibration set.

        Parameters
        ----------
        pred_calib : np.ndarray
            Predictions on the calibration set.
        y_calib : np.ndarray
            True labels/targets for the calibration set.
        pred_test : np.ndarray
            Predictions on the test set.
        y_test : np.ndarray
            True labels/targets for the test set (used for evaluation only).

        Returns
        -------
        pd.DataFrame
            DataFrame containing lower bounds, upper bounds, true test values, and a binary
            violation indicator (1 if y_test is outside the interval, else 0).
        """
        # Step 1: Compute absolute residuals (nonconformity scores) on the calibration set
        calib_scores = np.abs(pred_calib - y_calib)

        # Step 2: Sort the calibration scores
        calib_scores = np.sort(calib_scores)

        # Step 3: Estimate the quantile of interest (q̂), which defines the interval width
        # Using the 'higher' method ensures conservative coverage
        q_hat = np.quantile(calib_scores, 1 - self.alpha, method='higher')

        # Step 4: Construct prediction intervals around test predictions
        lower_bounds = pred_test - q_hat
        upper_bounds = pred_test + q_hat

        # Step 5: Check for violations — whether y_test is outside the prediction interval
        violations = ((y_test < lower_bounds) | (y_test > upper_bounds)).astype(int)

        # Step 6: Return a formatted DataFrame with results
        return self._make_final_dataframe(lower_bounds, upper_bounds, y_test, violations)

    # -------------------------------------------
    # 2. Conformalized Quantile Regression (CQR)
    # -------------------------------------------
    def cqr(self, pred_lower_calib, pred_upper_calib, y_calib, pred_lower_test, pred_upper_test, y_test):
        """
        Compute conformalized quantile regression (CQR) prediction intervals.

        CQR adjusts model-predicted quantile intervals by calibrating based on the 
        maximum deviation between observed calibration targets and predicted intervals.

        Parameters
        ----------
        pred_lower_calib : np.ndarray
            Predicted lower quantiles on the calibration set.
        pred_upper_calib : np.ndarray
            Predicted upper quantiles on the calibration set.
        y_calib : np.ndarray
            True labels/targets for the calibration set.
        pred_lower_test : np.ndarray
            Predicted lower quantiles on the test set.
        pred_upper_test : np.ndarray
            Predicted upper quantiles on the test set.
        y_test : np.ndarray
            True labels/targets for the test set (used for evaluation only).

        Returns
        -------
        pd.DataFrame
            DataFrame containing calibrated lower bounds, upper bounds, true test values,
            and a binary violation indicator (1 if y_test is outside the interval, else 0).
        """
        # Step 1: Compute nonconformity scores based on max distance from predicted interval
        calib_scores = np.maximum(pred_lower_calib - y_calib, y_calib - pred_upper_calib)

        # Step 2: Sort the calibration scores
        calib_scores = np.sort(calib_scores)

        # Step 3: Estimate the quantile of interest (q̂), which defines the interval adjustment
        q_hat = np.quantile(calib_scores, 1 - self.alpha, method='higher')

        # Step 4: Adjust test prediction intervals by q̂
        lower_bounds = pred_lower_test - q_hat
        upper_bounds = pred_upper_test + q_hat

        # Step 5: Check for violations — whether y_test is outside the adjusted prediction interval
        violations = ((y_test < lower_bounds) | (y_test > upper_bounds)).astype(int)

        # Step 6: Return a formatted DataFrame with results
        return self._make_final_dataframe(lower_bounds, upper_bounds, y_test, violations)


    # ---------------------------
    # 3. Adaptive Conformal Inference (ACI)
    # ---------------------------
    def aci(self, pred_calib, y_calib, pred_test, y_test, gamma=0.005):
        """
        Compute adaptive conformal prediction intervals using online conformal adjustment.

        This method adapts the interval size per test point based on previous prediction success, 
        increasing or decreasing the width dynamically via a learning rate (gamma). Especially useful 
        for non-stationary or time-ordered data.

        Parameters
        ----------
        pred_calib : np.ndarray
            Predictions on the calibration set.
        y_calib : np.ndarray
            True labels/targets for the calibration set.
        pred_test : np.ndarray
            Predictions on the test set.
        y_test : np.ndarray
            True labels/targets for the test set (in temporal or streaming order).
        gamma : float, optional
            Adaptation rate for alpha (default is 0.005).

        Returns
        -------
        pd.DataFrame
            DataFrame containing per-step lower/upper bounds, true test values, and binary 
            violation indicators (1 if y_test is outside the interval, else 0).
        """
        # Step 1: Compute nonconformity scores (absolute residuals) on the calibration set
        calib_scores = np.abs(pred_calib - y_calib)
        calib_scores = np.sort(calib_scores)

        # Step 2: Initialize adaptive alpha with global alpha
        adaptive_alpha = self.alpha

        # Ensure test arrays are NumPy arrays
        pred_test = np.array(pred_test)
        y_test = np.array(y_test)

        # Step 3: Initialize output lists
        lower_bounds = []
        upper_bounds = []
        y_trues = []
        violations = []

        # Step 4: Iterate through test points sequentially
        for i in range(len(pred_test)):
            # Step 4.1: Determine quantile for current adaptive_alpha
            q_hat = np.quantile(calib_scores, 1 - adaptive_alpha, method='higher')

            # Step 4.2: Construct prediction interval
            lower = pred_test[i] - q_hat
            upper = pred_test[i] + q_hat

            # Step 4.3: Determine if there's a violation
            violation = int((y_test[i] < lower) or (y_test[i] > upper))

            # Step 4.4: Update adaptive alpha based on observed outcome
            adaptive_alpha += gamma * (self.alpha - violation)

            # Step 4.5: Clip alpha to stay within valid bounds [0, 1]
            adaptive_alpha = np.clip(adaptive_alpha, 0, 1)

            # Step 4.6: Store results
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            y_trues.append(y_test[i])
            violations.append(violation)

        # Step 5: Return formatted DataFrame with results
        return self._make_final_dataframe(
            np.array(lower_bounds),
            np.array(upper_bounds),
            np.array(y_trues),
            np.array(violations)
        )

