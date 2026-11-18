"""R15: a_calibrated_tod."""

import json
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import median

Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
CEILING = 1_000_000_000
MOMENTUM_WEIGHT = 1.155247
ROBUST_CUTOFF = 0.959268
SLOPE_BANDWIDTH = 5.807776
SLOPE_MIX = 0.35959
SPREAD_WEIGHT = 0.433887
SKEW = -0.036039
TREND_WINDOW = 4
TREND_WEIGHT = 0.31931
POOL_THRESHOLD = 10057
TREND_MODE = 'add'
HOURS = range(2, 25)


def positive_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric viewer count")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Expected a positive viewer count")
    return number


def first_print(case):
    """Recover the anchor independently of failures in the training data."""
    if not isinstance(case, dict):
        return 1.0
    known = case.get("known", [])
    if isinstance(known, list):
        for point in known:
            if isinstance(point, dict) and point.get("hour") == 1:
                try:
                    return positive_number(point.get("viewers"))
                except (ValueError, TypeError, OverflowError):
                    pass
    history = case.get("history_hourly", [])
    if isinstance(history, list):
        for point in reversed(history):
            if not isinstance(point, dict):
                continue
            try:
                return positive_number(point.get("viewers"))
            except (ValueError, TypeError, OverflowError):
                pass
    return 1.0


def rounded_viewers(value):
    value = min(float(CEILING), max(0.01, value))
    scaled = value * 100
    whole = math.floor(scaled)
    cents = whole + (scaled - whole >= 0.5)
    return cents // 100 if cents % 100 == 0 else cents / 100


def flat_forecast(p1):
    value = min(float(CEILING), p1)
    value = int(value) if value.is_integer() else value
    return {"hours": [{"hour": h, "q": [value] * 5} for h in range(2, 25)]}


def utc_time(text):
    instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def launch_pool(comparables, *, with_prior=False, min_anchor=20000):
    if not isinstance(comparables, list):
        raise ValueError("Comparables must be a list")
    pool = []
    for launch in comparables:
        if not isinstance(launch, dict):
            continue
        if not isinstance(launch.get("unlock"), str):
            continue
        if not isinstance(launch.get("day_start"), str):
            continue
        if not isinstance(launch.get("hourly"), list):
            continue
        unlock = launch["unlock"].lower()
        if any(tag in unlock for tag in ("early-access", "target-history", "tier: weak")):
            continue
        try:
            start = utc_time(launch["day_start"])
        except (ValueError, TypeError, OverflowError):
            continue
        series = {}
        for point in launch["hourly"]:
            if not isinstance(point, dict) or not isinstance(point.get("from"), str):
                continue
            try:
                value = positive_number(point.get("viewers"))
                elapsed = (utc_time(point["from"]) - start).total_seconds()
            except (ValueError, TypeError, OverflowError):
                continue
            if 0 <= elapsed < 86400:
                hour = int(elapsed // 3600) + 1
                series.setdefault(hour, float(value))
        if series.get(1, 0) >= min_anchor and sum(h in series for h in HOURS) >= 20:
            if with_prior:
                prior = previous_hour(start, launch["hourly"])
                if prior is not None:
                    series[0] = prior
            trend = prelaunch_trend(start, launch["hourly"])
            if trend is not None:
                series[-1] = trend
            pool.append((math.log(series[1]), series))
    return pool


def least_squares(xs, ys, weights=None):
    weights = [1.0] * len(xs) if weights is None else weights
    total = math.fsum(weights)
    mean_x = math.fsum(w * x for w, x in zip(weights, xs)) / total
    mean_y = math.fsum(w * y for w, y in zip(weights, ys)) / total
    variance = math.fsum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    covariance = math.fsum(
        w * (x - mean_x) * (y - mean_y)
        for w, x, y in zip(weights, xs, ys)
    )
    slope = covariance / variance if variance > 1e-20 else 0.0
    return mean_y - slope * mean_x, slope


def residual_summary(xs, ys, intercept, slope):
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    midpoint = median(residuals)
    deviations = [abs(r - midpoint) for r in residuals]
    return midpoint, deviations, median(deviations)


def robust_hour(xs, ys):
    intercept, slope = least_squares(xs, ys)
    for _ in range(3):
        _, deviations, mad = residual_summary(xs, ys, intercept, slope)
        # Exact fits can have zero MAD; a numerical floor avoids infinite weights.
        cutoff = max(1e-12, ROBUST_CUTOFF * 1.4826 * mad)
        weights = [1.0 / max(d, cutoff) for d in deviations]
        intercept, slope = least_squares(xs, ys, weights)
    midpoint, deviations, mad = residual_summary(xs, ys, intercept, slope)
    return intercept, slope, midpoint, mad, deviations


def forecast_v2(pool, p1):
    if not pool:
        return flat_forecast(p1)
    centre = median([x for x, _ in pool]) if pool else 0.0
    case_level = math.log(p1) - centre
    fits = {}
    spread_x, spread_y = [], []
    for hour in range(2, 25):
        xs, ys = [], []
        for log_anchor, series in pool:
            if hour in series:
                xs.append(log_anchor - centre)
                ys.append(math.log(series[hour]) - log_anchor)
        if len(xs) < 6:
            continue
        intercept, slope, midpoint, mad, deviations = robust_hour(xs, ys)
        fits[hour] = (intercept + slope * case_level + midpoint, max(0.05, 1.4826 * mad) * 1.2)
        spread_x.extend(xs)
        spread_y.extend(math.log(max(d, 1e-12)) for d in deviations)
    spread_slope = least_squares(spread_x, spread_y)[1] if spread_x else 0.0
    factor = math.exp(max(-0.7, min(0.7, 0.5 * spread_slope * case_level)))
    rows = []
    for hour in range(2, 25):
        if hour in fits:
            location, spread = fits[hour]
            spread *= factor
        else:
            location, spread = 0.0, 0.15 * math.sqrt(hour - 1)
        quantiles = []
        for z in Z_SCORES:
            log_value = math.log(p1) + location + z * spread
            if log_value >= math.log(CEILING):
                value = CEILING
            elif log_value <= math.log(0.01):
                value = 0.01
            else:
                value = rounded_viewers(math.exp(log_value))
            quantiles.append(max(quantiles[-1], value) if quantiles else value)
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def smooth_hour(hour, values, bandwidth):
    neighbours = [(math.exp(-0.5 * ((hour - other) / bandwidth) ** 2), value)
                  for other, value in values.items()]
    return math.fsum(w * b for w, b in neighbours) / math.fsum(w for w, _ in neighbours)


def smoothed_profile(pool, log_p1):
    """Return per-hour log medians and robust scales after pooling slopes."""
    centre = median(x for x, _ in pool)
    target = log_p1 - centre
    hourly = {}
    for hour in HOURS:
        points = [(x - centre, math.log(series[hour]) - x)
                  for x, series in pool if hour in series]
        if len(points) >= 6:
            xs, ys = zip(*points)
            hourly[hour] = (xs, ys, robust_hour(xs, ys)[1])

    profile, spread_x, spread_y = {}, [], []
    slopes = {h: fit[2] for h, fit in hourly.items()}
    for hour, (xs, ys, raw_slope) in hourly.items():
        slope = (1.0 - SLOPE_MIX) * raw_slope + SLOPE_MIX * smooth_hour(hour, slopes, SLOPE_BANDWIDTH)
        # Recenter after shrinking the slope so location and spread stay consistent.
        intercept, deviations, mad = residual_summary(xs, ys, 0.0, slope)
        profile[hour] = (log_p1 + intercept + slope * target, max(0.05, 1.4826 * mad) * 1.2)
        spread_x.extend(xs)
        spread_y.extend(math.log(max(d, 1e-12)) for d in deviations)

    spread_slope = least_squares(spread_x, spread_y)[1] if spread_x else 0.0
    factor = math.exp(max(-0.7, min(0.7, 0.5 * spread_slope * target)))
    return {h: (loc, scale * factor) for h, (loc, scale) in profile.items()}


def previous_hour(start, history):
    """The only history feature is the hour ending at day_start."""
    if not isinstance(history, list):
        return None
    for point in reversed(history):
        if not isinstance(point, dict) or not isinstance(point.get("from"), str):
            continue
        try:
            offset = (utc_time(point["from"]) - start).total_seconds()
            if -3600 <= offset < 0:
                return positive_number(point.get("viewers"))
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def point_momentum_adjustments(pool, log_p1, log_prior):
    # Compare the boundary jump relative to the typical jump at this anchor.
    curves = [(x, x - math.log(series[0]), series) for x, series in pool if 0 in series]
    if len(curves) < 8:
        return {}
    intercept, slope = least_squares([x for x, _, _ in curves], [m for _, m, _ in curves])
    surprise = (1.0 - slope) * log_p1 - log_prior - intercept
    changes = {}
    for hour in HOURS:
        points = [(x, m - intercept - slope * x, math.log(series[hour]) - x)
                  for x, m, series in curves if hour in series]
        if len(points) < 6:
            continue
        xs, jumps, ratios = zip(*points)
        a, b = least_squares(xs, ratios)
        ratios_without_level = [y - a - b * x for x, y in zip(xs, ratios)]
        changes[hour] = robust_hour(jumps, ratios_without_level)[1]
    weight = MOMENTUM_WEIGHT * max(-2.0, min(2.0, surprise))
    return {h: weight * smooth_hour(h, changes, 4.0) for h in HOURS} if changes else {}


def prelaunch_trend(start, history):
    """OLS log-viewer slope per hour; require all N pre-boundary hourly bins."""
    if not isinstance(history, list):
        return None
    bins = {}
    for point in history:
        if not isinstance(point, dict) or not isinstance(point.get("from"), str):
            continue
        try:
            offset = (utc_time(point["from"]) - start).total_seconds() / 3600
            if -TREND_WINDOW <= offset < 0:
                bins.setdefault(math.floor(offset), (offset, math.log(positive_number(point.get("viewers")))))
        except (ValueError, TypeError, OverflowError):
            continue
    if len(bins) != TREND_WINDOW:
        return None
    points = [bins[k] for k in sorted(bins)]
    return least_squares([p[0] for p in points], [p[1] for p in points])[1]


def momentum_adjustments(pool, log_p1, log_prior, trend=None):
    point = point_momentum_adjustments(pool, log_p1, log_prior)
    curves = [(x, series[-1], series) for x, series in pool if -1 in series]
    if trend is None or len(curves) < 8:
        return point
    # Remove anchor-level dependence from both the trend and response.
    intercept, slope = least_squares([x for x, _, _ in curves], [t for _, t, _ in curves])
    surprise = trend - intercept - slope * log_p1
    changes = {}
    for hour in HOURS:
        points = [(x, t - intercept - slope * x, math.log(series[hour]) - x)
                  for x, t, series in curves if hour in series]
        if len(points) < 6:
            continue
        xs, trends, ratios = zip(*points)
        a, b = least_squares(xs, ratios)
        residuals = [y - a - b * x for x, y in zip(xs, ratios)]
        changes[hour] = robust_hour(trends, residuals)[1]
    if not changes:
        return point
    weight = TREND_WEIGHT * max(-2.0, min(2.0, surprise))
    return {h: (point.get(h, 0.0) if TREND_MODE == "add" else 0.0)
            + weight * smooth_hour(h, changes, 4.0) for h in HOURS}


def hourly_spread_factors(pool):
    """Smooth relative spread errors estimated by LOO inside the training pool."""
    if len(pool) < 9:
        return {}
    errors = {h: [] for h in HOURS}
    for index, (anchor, series) in enumerate(pool):
        inner = pool[:index] + pool[index + 1:]
        profile = smoothed_profile(inner, anchor)
        updates = momentum_adjustments(inner, anchor, math.log(series[0]), series.get(-1)) if 0 in series else {}
        for hour, (loc, scale) in profile.items():
            if hour in series:
                residual = math.log(series[hour]) - loc - updates.get(hour, 0.0)
                errors[hour].append(abs(residual) / scale)
    ratios = {h: max(0.1, median(values)) for h, values in errors.items() if values}
    if not ratios:
        return {}
    typical = median(ratios.values())
    log_ratios = {h: math.log(value / typical) for h, value in ratios.items()}
    return {h: math.exp(max(-0.4, min(0.4, SPREAD_WEIGHT * smooth_hour(h, log_ratios, 3.0)))) for h in HOURS}


def empirical_quantile(ordered, tau):
    position = tau * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (position - left) * (ordered[right] - ordered[left])


@lru_cache(maxsize=1)
def early_late_offsets(pool_key):
    """Estimate R9-style offsets on held-out training comparables."""
    offsets = [z + SKEW * z * z for z in Z_SCORES]
    if len(pool_key) < 10:
        return offsets, 0.0
    pool = [(x, dict(points)) for x, points in pool_key]
    early, late = [], []
    for index, (anchor, series) in enumerate(pool):
        inner = pool[:index] + pool[index + 1:]
        profile = smoothed_profile(inner, anchor)
        updates = (momentum_adjustments(inner, anchor, math.log(series[0]), series.get(-1))
                   if 0 in series else {})
        factors = hourly_spread_factors(inner)
        for hour, (location, spread) in profile.items():
            if hour not in series:
                continue
            residual = math.log(series[hour]) - location - updates.get(hour, 0.0)
            if hour < 7:
                early.append(residual / (spread * factors.get(hour, 1.0)))
            else:
                late.append(residual)

    # Keep the median unchanged in hours 2-6 and preserve quantile order.
    if early:
        early.sort()
        for k, tau in ((0, 0.05), (1, 0.25), (3, 0.75), (4, 0.95)):
            offsets[k] = 0.5 * offsets[k] + 0.5 * empirical_quantile(early, tau)
        offsets[0] = min(offsets[0], offsets[1], 0.0)
        offsets[1] = min(0.0, max(offsets[0], offsets[1]))
        offsets[3] = max(0.0, offsets[3])
        offsets[4] = max(offsets[3], offsets[4])
    return offsets, 0.5 * median(late) if late else 0.0


def tod_offset(comparables, day_start):
    """R11 UTC hour and weekday shrinkage (n / (n + 24))."""
    if not isinstance(day_start, str):
        return 0.0
    try:
        start = utc_time(day_start)
    except (ValueError, OverflowError):
        return 0.0
    observations = []
    for launch in comparables:
        if not isinstance(launch, dict) or not isinstance(launch.get("day_start"), str):
            continue
        try:
            dt = utc_time(launch["day_start"])
            one = launch_pool([launch], with_prior=True, min_anchor=10000)
            if one:
                x, series = one[0]
                levels = [math.log(series[h]) - x for h in HOURS if h in series]
                if levels:
                    observations.append((dt.hour, dt.weekday(), median(levels)))
        except (ValueError, TypeError, OverflowError, KeyError):
            continue
    if observations:
        global_level = median(value for _, _, value in observations)
        hour_values = [value for hour, _, value in observations if hour == start.hour]
        day_values = [value for _, day, value in observations if day == start.weekday()]
        hour_offset = ((median(hour_values) - global_level) * len(hour_values) / (len(hour_values) + 24.0)
                       if len(hour_values) >= 2 else 0.0)
        day_offset = ((median(day_values) - global_level) * len(day_values) / (len(day_values) + 24.0)
                      if len(day_values) >= 2 else 0.0)
        return max(-0.5, min(0.5, hour_offset + day_offset))
    return 0.0


def forecast(case, p1):
    comparables = case["comparables"]
    day_start = case.get("day_start")
    prior = None
    if isinstance(day_start, str):
        try:
            prior = previous_hour(utc_time(day_start), case.get("history_hourly"))
        except (ValueError, OverflowError):
            pass
    # Expand forecast training only. The default pool still defines the LOO cohort.
    strict = launch_pool(comparables, with_prior=prior is not None, min_anchor=POOL_THRESHOLD)
    if len(strict) < 6:
        return forecast_v2(strict, p1)
    try:
        log_p1 = math.log(p1)
        profile = smoothed_profile(strict, log_p1)
        if any(h not in profile for h in HOURS):
            return forecast_v2(strict, p1)
        updates = momentum_adjustments(strict, log_p1, math.log(prior),
                                       prelaunch_trend(utc_time(day_start), case.get("history_hourly"))) if prior is not None else {}
        factors = hourly_spread_factors(strict)
        nominal = [z + SKEW * z * z for z in Z_SCORES]
        early, late_bias = nominal, 0.0
        key = tuple((anchor, tuple(series.items())) for anchor, series in strict)
        early, late_bias = early_late_offsets(key)
        clock_offset = tod_offset(comparables, day_start)
        low, high = math.log(0.01), math.log(CEILING)
        rows = []
        for hour in HOURS:
            location, spread = profile[hour]
            spread *= factors.get(hour, 1.0)
            location += updates.get(hour, 0.0) + clock_offset
            offsets = early if hour < 7 else nominal
            if hour >= 7:
                location += late_bias
            qs = [rounded_viewers(math.exp(max(low, min(high, location + z * spread))))
                  for z in offsets]
            rows.append({"hour": hour, "q": qs})
        return {"hours": rows}
    except (ArithmeticError, ValueError):
        return forecast_v2(strict, p1)


def process_file(source, output_dir):
    p1 = 1.0
    try:
        case = json.loads(source.read_text(encoding="utf-8"))
        p1 = first_print(case)
        result = forecast(case, p1)
        payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
        if len(payload.encode("utf-8")) > 65536:
            raise ValueError("Forecast exceeds output limit")
    except Exception:
        payload = json.dumps(flat_forecast(p1), separators=(",", ":"), allow_nan=False)
    target = output_dir / (source.stem + ".out")
    try:
        target.write_text(payload, encoding="utf-8")
    except Exception:
        # Retry with the smallest valid forecast after a transient write failure.
        try:
            target.write_text(json.dumps(flat_forecast(p1), separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass


def main():
    try:
        input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
        output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(input_dir.glob("*.in")):
            process_file(source, output_dir)
    except Exception:
        # There is no recoverable output target when directory discovery fails.
        pass


if __name__ == "__main__":
    main()
