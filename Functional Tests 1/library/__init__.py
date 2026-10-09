"""
Python port of the FinTech-545-style `library/` used by testfiles/test_setup.jl.

Ported modules
---------------
missing_cov.py       -> missing_cov()
ew_cov.py             -> ew_covar(), exp_weights()
psd.py                -> near_psd(), chol_psd(), higham_nearest_psd()
simulate.py           -> simulate_normal(), simulate_pca()
return_calculate.py   -> return_calculate()
fitted_model.py       -> FittedModel, aicc(), fit_normal(), fit_general_t(),
                          fit_regression_t(), fit_nig_moments(), fit_NIG_mle()

Only the pieces that testfiles/test_setup.jl actually calls were ported (the
Julia project also has bt_american.jl, gbsm.jl, RiskStats.jl, copula.jl,
multivariate_t.jl, optimizers.jl, expost_factor.jl, skewNormal.jl,
johnsonsu.jl, fit_regression_nig.jl -- none of those are exercised by
test_setup.jl, so they were left untranslated).

IMPORTANT -- about randomness
------------------------------
Every function here that is *deterministic given its input* (missing_cov,
ew_covar, near_psd, chol_psd, higham_nearest_psd, return_calculate, and all
of the fit_* functions) reproduces the Julia algorithm step for step and
will match the Julia output to floating point precision when run on the
*same input data*.

Functions that draw random numbers (simulate_normal, simulate_pca, and the
input-generation blocks in testfiles/test_setup.py) use NumPy's RNG, which
does **not** produce the same stream as Julia's `Random`/`Distributions`.
So the *inputs* generated fresh by Random.seed!(...) in Julia cannot be
reproduced bit-for-bit from Python, and neither can outputs that are sample
statistics of a fresh random draw (e.g. testout_5.*, the covariance of a
100,000-row simulateNormal draw). If you already have the data/*.csv input
files that the Julia script produced, load those instead of regenerating,
and everything downstream (the deterministic transforms) will line up.
"""
