"""
qrmlib -- reusable building blocks for FinTech 545 (Quantitative Risk Management).

Modules
-------
missing      : which cells are observed; joint-observation counts
correlation  : complete-case and pairwise covariance/correlation, eigenvalue helpers
cholesky     : hand-written Cholesky (PD and PSD) with a non-raising attempt wrapper
psd_repair   : Rebonato-Jackel and Higham repairs of a non-PSD correlation matrix
portfolio    : covariance from correlations + std devs, portfolio variance
returns      : arithmetic / log returns from prices, demeaning
ewma         : exponentially weighted variance, effective sample size, half life
moments      : first four sample moments, normality diagnostics
fitting      : normal and Student t maximum likelihood fits
var          : normal, t and historical VaR; historical and normal ES
tails        : tail-event counts across scenarios, subadditivity check
regime       : variance break point, normal-mixture kurtosis
copula       : margins by AICc, Kendall R, Gaussian and t copulas, joint tails
factor       : single-index OLS model, residual covariance, factor simulation
"""
from .missing import (observed_mask, joint_obs_counts, complete_rows,
                      n_complete, missing_summary)
from .correlation import (cov_complete, corr_complete, cov_pairwise, corr_pairwise,
                          eigenvalues, min_eigenvalue, is_psd)
from .cholesky import (CholeskyError, CholeskyResult, chol_pd, chol_psd,
                       attempt_cholesky)
from .psd_repair import (HighamResult, frobenius_distance, rebonato_jackel,
                         higham_nearest_correlation)
from .portfolio import (full_history_std, cov_from_corr, portfolio_variance,
                        tracking_weights, scenario_pnl)

from .returns import return_calculate, demean
from .ewma import (ew_weights, ew_covariance, ew_variance, ew_std,
                   effective_sample_size, half_life, weight_on_recent, ew_std_error)
from .moments import four_moments, skew_kurt_std_errors, jarque_bera, extreme_influence
from .fitting import TFit, fit_normal, fit_t
from .var import (var_normal, var_t, var_historical, var_noise_normal,
                  es_historical, es_normal)
from .tails import tail_counts, subadditivity
from .regime import variance_breakpoint, regime_stds, normal_mixture_moments

from .copula import (aicc, bic, MarginFit, fit_margin, to_uniforms, from_uniforms,
                     pseudo_uniforms, joint_tail_counts, expected_joint_tail_independent,
                     kendall_corr, gaussian_copula_logpdf, t_copula_logpdf, CopulaFit,
                     fit_gaussian_copula, fit_t_copula, simulate_gaussian_copula,
                     simulate_t_copula, t_tail_dependence, gaussian_tail_dependence)
from .factor import OLSFit, ols, FactorModel, fit_factor_model

__version__ = "0.6.0"
