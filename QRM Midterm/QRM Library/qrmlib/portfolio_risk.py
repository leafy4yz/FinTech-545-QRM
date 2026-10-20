"""
qrmlib.portfolio_risk -- $ VaR and ES for a portfolio of stocks (Weeks 4 and 5).

Positions are linear, so with arithmetic returns  P&L_i = value_i * r_i  and the
portfolio P&L is the row sum.  Every function returns a table with one row per
stock plus a "Portfolio" row, columns VaR and ES (positive = loss).

    shares = shares_for_weights(prices.iloc[-1], 100_000)        # equal weights
    values = position_values(shares, prices.iloc[-1])
    hist   = historical_risk(rets, values, alpha=0.01)
    dn     = delta_normal_risk(values, rets.cov(), alpha=0.01)
    gc, info = copula_risk(rets, values, alpha=0.01, family="t", rng=545)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .copula import (fit_margins, margin_table, to_uniforms, from_uniforms, kendall_corr,
                     spearman_corr, fit_t_copula, simulate_gaussian_copula,
                     simulate_t_copula)
from .simulation import simulate_normal
from .var import var_es_table


def shares_for_weights(prices, total_value: float, weights=None) -> pd.Series:
    """
    Shares (fractions allowed) so position i is worth weights_i * total_value at
    the given prices.  prices: Series of current prices (e.g. prices.iloc[-1]
    with any date column dropped).  weights=None -> equal weights.
    """
    p = pd.Series(prices, dtype=float)
    w = (pd.Series(1.0 / len(p), index=p.index) if weights is None
         else pd.Series(weights, index=p.index, dtype=float))
    return (w * total_value / p).rename("shares")


def position_values(shares, prices) -> pd.Series:
    """$ value of each position: shares * price."""
    return (pd.Series(shares, dtype=float) * pd.Series(prices, dtype=float)).rename("value")


def pnl_from_returns(returns: pd.DataFrame, values) -> pd.DataFrame:
    """$ P&L per position for every scenario row: value_i * r_i."""
    v = pd.Series(values, dtype=float)
    return pd.DataFrame(returns)[v.index].mul(v, axis=1)


def historical_risk(returns: pd.DataFrame, values, alpha: float = 0.05,
                    method: str = "class", total: str = "Portfolio") -> pd.DataFrame:
    """Historical simulation: every historical return row applied to today's positions."""
    return var_es_table(pnl_from_returns(returns, values), alpha, method, total)


def delta_normal_risk(values, cov, alpha: float = 0.05,
                      total: str = "Portfolio") -> pd.DataFrame:
    """
    Closed-form normal VaR / ES with zero mean (Week 4, Section 4; Week 5, 1.1):
        sigma_$ = sqrt(v' Sigma v),  VaR = -z_alpha sigma_$,  ES = phi(z_alpha)/alpha sigma_$
    """
    v = pd.Series(values, dtype=float)
    s = pd.DataFrame(cov).loc[v.index, v.index].to_numpy(float)
    z = stats.norm.ppf(alpha)
    sd = np.append(np.sqrt(np.diag(s)) * np.abs(v.to_numpy()),
                   np.sqrt(v.to_numpy() @ s @ v.to_numpy()))
    return pd.DataFrame({"VaR": -z * sd, "ES": stats.norm.pdf(z) / alpha * sd},
                        index=list(v.index) + [total])


def normal_mc_risk(values, cov, alpha: float = 0.05, n_sim: int = 100_000, rng=None,
                   method: str = "class", total: str = "Portfolio"):
    """Monte Carlo with multivariate normal returns, zero mean.  Returns (table, sims)."""
    v = pd.Series(values, dtype=float)
    sims = simulate_normal(pd.DataFrame(cov).loc[v.index, v.index], n_sim, rng=rng)
    return var_es_table(pnl_from_returns(sims, v), alpha, method, total), sims


def copula_risk(returns: pd.DataFrame, values, alpha: float = 0.05, family="t",
                copula: str = "gaussian", corr: str | None = None,
                n_sim: int = 100_000, rng=None, method: str = "class",
                total: str = "Portfolio", normal_sd: str = "sample"):
    """
    Copula Monte Carlo VaR / ES (Week 5, Sections 4 and 7):

      1. fit a margin to each stock     (family: "t", "normal", "aicc" or a dict;
                                         normal margins use the n - 1 sd unless
                                         normal_sd="mle")
      2. U = F_i(r_i)                   through each fitted CDF
      3. R from U                       (corr: "spearman" or "kendall";
                                         default spearman for gaussian, kendall for t)
      4. simulate copula uniforms       (copula: "gaussian" or "t", nu by MLE)
      5. r_i = F_i^{-1}(U_i), P&L = value * r, VaR / ES per stock and portfolio

    Returns (table, info); info holds margins, params (margin table), R, copula
    fit (t only), sims (simulated returns) and pnl.
    """
    v = pd.Series(values, dtype=float)
    rets = pd.DataFrame(returns)[v.index]
    margins = fit_margins(rets, family, normal_sd)
    u = to_uniforms(rets, margins)
    corr = corr or ("spearman" if copula == "gaussian" else "kendall")
    if corr == "spearman":
        r = spearman_corr(u)
    elif corr == "kendall":
        r = kendall_corr(u)
    else:
        raise ValueError("corr must be 'spearman' or 'kendall'")
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    cfit = None
    if copula == "gaussian":
        usim = simulate_gaussian_copula(r, n_sim, g)
    elif copula == "t":
        cfit = fit_t_copula(u, r)
        usim = simulate_t_copula(r, cfit.nu, n_sim, g)
    else:
        raise ValueError("copula must be 'gaussian' or 't'")
    sims = from_uniforms(usim, margins, list(v.index))
    pnl = pnl_from_returns(sims, v)
    table = var_es_table(pnl, alpha, method, total)
    info = {"margins": margins, "params": margin_table(margins),
            "R": pd.DataFrame(r, index=v.index, columns=v.index),
            "copula_fit": cfit, "sims": sims, "pnl": pnl}
    return table, info


def group_pnl(pnl: pd.DataFrame, groups, total: str = "Total") -> pd.DataFrame:
    """
    Add one P&L column per group (e.g. book) and a total column to a per-position
    P&L table.  groups: Series mapping position -> group name.
    Column order: positions, groups (sorted), total.
    """
    g = pd.Series(groups)
    out = pd.DataFrame(pnl).copy()
    pos = list(out.columns)
    for name in sorted(g.unique()):
        members = [c for c in pos if g.get(c) == name]
        out[f"Book {name}"] = out[members].sum(axis=1)
    out[total] = out[pos].sum(axis=1)
    return out


def compare_methods(tables: dict) -> pd.DataFrame:
    """
    Stack several VaR/ES tables side by side, e.g.
        compare_methods({"Gaussian copula": gc, "Historical": hist, "Delta normal": dn})
    Columns become a (method, VaR/ES) MultiIndex.
    """
    return pd.concat(tables, axis=1)
