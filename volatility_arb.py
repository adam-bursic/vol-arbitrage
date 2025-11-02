"""Volatility strategy module"""

import pandas as pd
import numpy as np
import QuantLib as ql
from arch import arch_model


__all__ = [
    "variance_forecast",
    "rolling_percentile",
    "regime_indicator",
    "momentum_indicator",
    "term_struc_indicator",
    "volscore",
    "short_straddle_calc",
    "short_straddle_path",
    "short_straddle_strategy_delta_hedged",
]


def variance_forecast(
    rets: pd.Series,
    start_date,
    end_date,
    *,
    vol="GARCH",
    mean="Constant",
    dist="Normal",
    p=1, q=1, o=0, power=2.0,
    horizon=21,
    lookback=756,
    refit_every=21,
    method="analytic",
    annualization=252.0,
    min_obs=100,
    rescale=True,
) -> pd.Series:
    """
    OOS forecast of cumulative, annualized variance over `horizon` days based on arch library
    
    :param rets: daily returns in decimals
    :param vol: 'GARCH' | 'ARCH' | 'EGARCH' | 'FIGARCH' | 'APARCH' | 'HARCH'
    :param mean: 'Constant' | 'Zero' | 'LS' | 'AR' | 'ARX' | 'HAR' | 'HARX' | 'constant'
    :param dist: 'normal' | 'gaussian' | 't' | 'studentst' | 'skewstudent' | 'skewt' | 'ged' | 'generalized error'
    :param method: 'analytic' | 'simulation' | 'bootstrap'
    :param horizon: length of step-ahead variance path
    :param lookback: window to estimate parameters
    :param refit_every: how often to refit the parameters
    :return: realized variance estimate
    
    """
    # clean, reindex
    r = rets.dropna().copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    if r.abs().max() > 1:
        raise ValueError("Returns must be decimals (0.02 for 2%)")
    mask = (r.index >= pd.Timestamp(start_date)) & (r.index <= pd.Timestamp(end_date))
    origins_idx = r.index[mask]
    if origins_idx.empty:
        return pd.Series(dtype=float, name=f"exp_var_{horizon}d_ann")
    
    # fit model and output
    am = arch_model(r, mean=mean, vol=vol, p=p, o=o, q=q, power=power, dist=dist, rescale=rescale)
    out = pd.Series(index=origins_idx, dtype=float, name=f"exp_var_{horizon}d")

    # walk through, refitting model at each refit_every
    index = r.index
    pos_lookup = pd.Series(np.arange(len(index)), index=index)
    i = 0
    n = len(origins_idx)
    while i < n:
        block_start_dt = origins_idx[i]
        last_obs_pos = int(pos_lookup.loc[block_start_dt])
        if lookback is None:
            first_obs_pos = 0
        else:
            first_obs_pos = max(0, last_obs_pos - lookback)
        if last_obs_pos - first_obs_pos < min_obs:
            i += refit_every
            continue
        res = am.fit(first_obs=first_obs_pos, last_obs=last_obs_pos, disp="off")
        f = res.forecast(horizon=horizon, method=method, align="origin")
        V = f.variance
        block_rows = V.reindex(origins_idx[i: min(i + refit_every, n)])
        block_rows = block_rows.dropna(how="all")
        if not block_rows.empty:
            cum_h = block_rows.sum(axis=1)
            ann = cum_h * (annualization / float(horizon))
            out.loc[ann.index] = ann.values
        i += refit_every

    return out / 10000


def rolling_percentile(s, win=126):
    """rank of last obs within window"""
    return s.rolling(win).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def regime_indicator(
    implied_vol: pd.Series, 
    lookback = 756, 
    hi_percentile = 80,
    smooth_window = 5,
    threshold = 0.5,
    ):
    """regime indicator based on implied vol"""
    p_hi = implied_vol.rolling(lookback, min_periods=252).quantile(hi_percentile / 100).shift(1)
    HighVolRisk = (implied_vol > p_hi).astype(float)
    HighVolRisk[p_hi.isna()] = np.nan

    HighVolRisk_smoothed = (
        HighVolRisk.rolling(smooth_window, min_periods=1)
        .mean()
        .gt(threshold)
        .astype(float)
    )
    regime_scale = (1.0 * (HighVolRisk_smoothed == 0)) + (0.0 * (HighVolRisk_smoothed == 1))
    regime_scale[p_hi.isna()] = np.nan

    return regime_scale


def momentum_indicator(
    implied_vol: pd.Series,
    window=63,
    ):
    """momentum indicator based on implied vol"""
    imp_vol = np.log(implied_vol.astype(float))
    vol_sma_long = imp_vol.rolling(window).mean().shift(1)
    VolMomentum = imp_vol - vol_sma_long

    VolMom_z = ((VolMomentum - VolMomentum.rolling(252).mean().shift(1)) /
            VolMomentum.rolling(252).std().shift(1)).clip(-5, 5)

    HighMomRisk = (VolMom_z > 0).astype(int)
    mom_scale = (1 - HighMomRisk)
    
    return mom_scale


def term_struc_indicator(
    implied_vol_1m,
    implied_vol_3m,
    window = 21,
    ):
    """ranking of the term structure profile within window"""
    iv1_var = (implied_vol_1m ** 2)
    iv3_var = (implied_vol_3m ** 2)
    term_slope = (iv3_var - iv1_var).shift(1)

    ts_ma = term_slope.rolling(window).mean()
    ts_z  = ((ts_ma - ts_ma.rolling(252).mean().shift(1)) /
         ts_ma.rolling(252).std().shift(1)).clip(-5, 5)

    def slope_scale(z, thr=0.0):
        return 0.5 * (1 + np.tanh((z - thr) / 2))

    ts_scale = slope_scale(ts_z, thr=0.0)
    
    return ts_scale


def volscore(
    implied_vol_1m: pd.Series,
    implied_vol_3m: pd.Series,
    spread: pd.Series,
    *,
    w_spread: float = 0.50,
    w_term: float = 0.50,
    spread_win: int = 126,
    regime_lookback: int = 756,
    regime_hi_pct: int = 80,
    regime_smooth_window: int = 5,
    regime_threshold: float = 0.50,
    term_window: int = 21,
    dropna: bool = True,
) -> pd.Series:
    """
    wrapper to compute volscore = (w_spread * spread_score + w_term * term_score) * regime_score
    """
    spread_score = rolling_percentile(spread.astype(float), win=spread_win).clip(0, 1)

    regime_score = regime_indicator(
        implied_vol_1m.astype(float),
        lookback=regime_lookback,
        hi_percentile=regime_hi_pct,
        smooth_window=regime_smooth_window,
        threshold=regime_threshold,
    )

    term_score = term_struc_indicator(
        implied_vol_1m.astype(float),
        implied_vol_3m.astype(float),
        window=term_window,
    )

    scores = pd.concat(
        {"spread": spread_score, "term": term_score, "regime": regime_score},
        axis=1,
    )

    volscore_series = (w_spread * scores["spread"] + w_term * scores["term"]) * scores["regime"]

    return volscore_series.dropna() if dropna else volscore_series


def short_straddle_calc(initiation, DTE, strike, spot, IV, divd, rf, market='NullCalendar'):
    """
    calculates the short straddle value and greeks
    :param initiation: date as YYYY-MM-DD
    :param DTE: days to expiration
    :param market: country calender, defaults to NullCalendar
    :return: ['Value', 'Delta', 'Gamma', 'Vega', 'Theta']
    """

    trade_date = ql.Date(initiation, '%Y-%m-%d')
    ql.Settings.instance().evaluationDate = trade_date
    try:
        calendar = getattr(ql, market)()
    except AttributeError:
        raise ValueError(f"Unknown market calendar: {market}")

    expiry = trade_date + DTE
    strike_price = strike
    call = ql.Option.Call
    put = ql.Option.Put
    spot_price = spot
    volatility = IV
    dividend_rate = divd
    risk_free_rate = rf

    call_payoff = ql.PlainVanillaPayoff(call, strike_price)
    put_payoff = ql.PlainVanillaPayoff(put, strike_price)
    
    exercise = ql.EuropeanExercise(expiry)
    call_option = ql.VanillaOption(call_payoff, exercise)
    put_option = ql.VanillaOption(put_payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    volatility_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(trade_date, calendar, ql.QuoteHandle(ql.SimpleQuote(volatility)), ql.Actual365Fixed()))
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(trade_date, ql.QuoteHandle(ql.SimpleQuote(dividend_rate)), ql.Actual365Fixed()))
    risk_free_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(trade_date, ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual365Fixed()))

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, risk_free_handle, volatility_handle)

    call_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    put_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    call_price = call_option.NPV()
    put_price = put_option.NPV()
    call_delta = call_option.delta()
    put_delta = put_option.delta()
    call_gamma = call_option.gamma()
    put_gamma = put_option.gamma()
    call_vega = call_option.vega()
    put_vega = put_option.vega()
    call_theta = call_option.theta()
    put_theta = put_option.theta()
    
    straddle_value = (call_price + put_price) * 100
    straddle_delta = -(call_delta + put_delta)
    straddle_gamma = -(call_gamma + put_gamma)
    straddle_vega = -(call_vega + put_vega) / 100
    straddle_theta = -(call_theta + put_theta) / 100

    return pd.Series(
        {"Value": straddle_value,
         "Delta": straddle_delta,
         "Gamma": straddle_gamma,
         "Vega":  straddle_vega,
         "Theta": straddle_theta
        })


def short_straddle_path(
    spot: pd.Series,
    implied_vol: pd.Series,
    riskfree_rate: pd.Series,
    dividends: pd.Series = None,
    market="NullCalendar"):
    """
    daily path for a monthly short straddle
    
    :param market: country calender, defaults to NullCalendar
    :return: ['DTE','Strike','Spot','Value','Delta','Gamma','Vega','Theta']
    """
    df = pd.concat([spot, implied_vol, riskfree_rate], axis=1)
    df.columns = ["spot", "iv", "rf"]
    if dividends is not None:
        df["divd"] = dividends
    else:
        df["divd"] = 0.0

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # monthly calc
    month_key = df.index.to_period("M")
    records = []
    for _, w in df.groupby(month_key):
        if len(w) < 1:
            continue
        end_dt   = w.index[-1]
        strike_locked = float(w["spot"].iloc[0])
        
        # dte = days in month
        for dt, row in w.iterrows():
            dte  = (end_dt - dt).days
            spot_t = float(row["spot"])
            iv_t   = float(row["iv"])
            rf_t   = float(row["rf"])
            div_t  = float(row["divd"])
            
            if dte == 0:
                val = abs(spot_t - strike_locked) * 100
                s = {
                    "Value": val,
                    "Delta": 0.0,
                    "Gamma": 0.0,
                    "Vega":  0.0,
                    "Theta": 0.0,
                }
            else:
                s = short_straddle_calc(
                    initiation = str(pd.Timestamp(dt).date()),
                    DTE        = int(dte),
                    strike     = strike_locked,
                    spot       = spot_t,
                    IV         = iv_t,
                    divd       = div_t,
                    rf         = rf_t,
                    market     = market,
                ).to_dict()

            s.update({"DTE": int(dte), "Strike": strike_locked, "Spot": spot_t})
            records.append((dt, s))

    out = pd.DataFrame({dt: vals for dt, vals in records}).T.sort_index()
    cols = ["DTE", "Strike", "Spot", "Value", "Delta", "Gamma", "Vega", "Theta"]
    out = out[cols]
    out.index.name = "Date"
    return out


def short_straddle_strategy_delta_hedged(
    path_df: pd.DataFrame,
    volscore: pd.Series,
    data_rf: pd.Series,
    nav_start: float = 100_000_000,
    contract_multiplier: int = 100,
    value_is_per_contract: bool = True,
    rf_mode: str = "daily"
) -> pd.DataFrame:
    """
    monthly-sized short straddle with daily mark-to-market and daily delta hedge.

    :param value_is_per_contract: if value is per point, set False (it will be × multiplier)
    :param rf_mode: 'daily' or 'monthly_first'
    :return:
    ['NAV','Cash','HedgeQty','HedgeTrade','HedgeValue','Liability','Contracts',
     'Strike','Spot','DTE','Value_contract','Delta_per_contract','Volscore',
     'rf_daily','MonthStart']
    """
    df_path = path_df.sort_index().copy()
    if not isinstance(df_path.index, pd.DatetimeIndex):
        df_path.index = pd.to_datetime(df_path.index)
    df_path.index = df_path.index.tz_localize(None)

    required_cols = {'Strike','Spot','Value','Delta'}
    missing = required_cols - set(df_path.columns)
    if missing:
        raise ValueError(f"path_df missing required columns: {missing}")

    for c in ['Strike','Spot','Value','Delta']:
        df_path[c] = pd.to_numeric(df_path[c], errors='coerce')

    if not value_is_per_contract:
        df_path = df_path.copy()
        df_path['Value'] = df_path['Value'] * contract_multiplier

    vol = volscore.sort_index().copy()
    if not isinstance(vol.index, pd.DatetimeIndex):
        vol.index = pd.to_datetime(vol.index)
    vol.index = vol.index.tz_localize(None)

    rf = data_rf.sort_index().copy()
    if not isinstance(rf.index, pd.DatetimeIndex):
        rf.index = pd.to_datetime(rf.index)
    rf.index = rf.index.tz_localize(None)
    rf = rf.reindex(df_path.index).ffill().fillna(0.0)

    # earliest day in each month where Strike & Value exist
    grp = df_path.assign(valid=df_path[['Strike','Value']].notna().all(axis=1)) \
                 .groupby(df_path.index.to_period("M"))
    # earliest valid index per month
    idx_first_valid = grp.apply(lambda g: g.index[g['valid']].min() if g['valid'].any() else pd.NaT)
    idx_first_valid = idx_first_valid.dropna().astype('datetime64[ns]')

    valid_first = df_path.loc[idx_first_valid, ['Strike','Value']]
    valid_first.index = pd.to_datetime(valid_first.index).tz_localize(None)

    # volscore
    vol_near = vol.reindex(valid_first.index, method="nearest")
    vol_near = vol_near.bfill().ffill()
    sizing_df = valid_first.join(vol_near.rename("Volscore")) 
    # check
    if sizing_df['Volscore'].isna().any():
        lost = sizing_df.index[sizing_df['Volscore'].isna()].to_period('M').astype(str).unique().tolist()
        raise RuntimeError(f"Volscore alignment produced NaN for months: {lost}")

    # daily MTM + daily delta hedge
    out = []
    NAV_prev  = float(nav_start)
    Q_prev    = 0.0
    Cash_prev = NAV_prev
    prev_dt   = None

    for start_dt, row in sizing_df.sort_index().iterrows():
        per = start_dt.to_period("M")
        month_data = df_path[df_path.index.to_period("M") == per]
        if month_data.empty:
            continue
        end_dt = month_data.index[-1]

        vol_w   = float(row["Volscore"])
        strike  = float(row["Strike"])
        v0      = float(row["Value"])
        denom   = strike * contract_multiplier
        if not np.isfinite(denom) or denom == 0:
            continue

        notional  = NAV_prev * vol_w
        contracts = (notional / denom)

        premium0  = contracts * v0
        Cash_prev = NAV_prev + premium0

        # rf locked to first day of month
        if rf_mode == "monthly_first":
            idx = rf.index.get_indexer([start_dt], method="nearest")[0]
            rf_ann_month = float(rf.iloc[idx])

        prev_dt = start_dt

        # walk trading days of  month
        for dt, mrow in month_data.iterrows():

            # cash accrue by rf
            day_frac = (dt - prev_dt).days / 365.0 if (prev_dt is not None and dt != prev_dt) else 0.0
            rf_ann   = float(rf.loc[dt]) if rf_mode == "daily" else rf_ann_month
            accrual  = (1.0 + rf_ann)**day_frac - 1.0
            Cash_today = Cash_prev * (1.0 + accrual)

            # per-contract option value and delta
            spot = float(mrow["Spot"])
            if dt == end_dt:
                # expiry = intrinsic per contract
                points = abs(spot - strike)
                v_today = points * contract_multiplier
                delta_per_contract = 0.0
            else:
                v_today = float(mrow["Value"])
                delta_per_contract = float(mrow["Delta"])

            # target hedge quantity
            # neutral = - delta_straddle * contracts * multiplier
            Q_target = - delta_per_contract * contracts * contract_multiplier

            # hedge trade at spot
            dQ = Q_target - Q_prev
            hedge_trade_cash = - dQ * spot
            Cash_today += hedge_trade_cash

            # liability and NAV calc
            Liability_today   = contracts * v_today
            HedgeValue_today  = Q_target * spot
            NAV_today         = Cash_today + HedgeValue_today - Liability_today

            # record
            dte = (end_dt - dt).days
            out.append({
                "Date": dt,
                "NAV": NAV_today,
                "Cash": Cash_today,
                "HedgeQty": Q_target,
                "HedgeTrade": dQ,
                "HedgeValue": HedgeValue_today,
                "Liability": Liability_today,
                "Contracts": contracts,
                "Strike": strike,
                "Spot": spot,
                "DTE": int(dte),
                "Value_contract": v_today,
                "Delta_per_contract": delta_per_contract,
                "Volscore": vol_w,
                "rf_daily": rf_ann,
                "MonthStart": start_dt
            })

            # if expiry day: close the hedge
            if dt == end_dt:
                close_cash = + Q_target * spot
                Cash_today += close_cash
                Q_target = 0.0
                HedgeValue_today = 0.0
                NAV_today = Cash_today + HedgeValue_today - Liability_today

            # advance state
            Cash_prev = Cash_today
            Q_prev    = Q_target
            NAV_prev  = NAV_today
            prev_dt   = dt

    out = pd.DataFrame(out).set_index("Date").sort_index()
    return out

