"""Forecast hourly Twitch concurrent-viewer counts for a game launch day.

The contest scores a pinball (quantile) loss on ln(viewers) at levels
0.05, 0.25, 0.5, 0.75 and 0.95.  For a pinball loss at level tau the
minimising action is the tau-quantile of the predictive distribution, so a
point estimate is the wrong output: this program emits five quantiles of a
predictive distribution over ln(viewers) for each hour 2..24.

Why the curve needs two locators.  A launch's audience is a function of hours
since its unlock, not of the wall-clock hour.  Hour 1's print alone does not
say where on that curve the launch sits: a title still waiting on a rolling
midnight unlock prints roughly its pre-launch idle level, while one that
unlocked hours before the checkpoint is already sliding off its peak.  So each
curve is described by three numbers: `base` (the pre-launch idle level, the
median of hourly read means over the 48 hours before the release day), `p1`
(hour 1's print) and `lift = ln(p1 / base)`.  The lift says how far hour 1
stands above idle, i.e. how much of the launch spike hour 1 has already seen.

The predictive distribution is then fitted at run time, per hour, by ordinary
least squares of ln(viewers) on the features [1.0, ln(p1), lift], trained on
the case's own `comparables` (past launches).  Curves whose hour 1 is not the
onset the rest of the day continues from -- early-access or staggered-unlock
launches -- are gated out by |lift| > 6.0, as are thin curves with fewer than
20 populated release-day hours.  The regression residuals carry the shape of
the uncertainty: their empirical quantiles, blended 40/60 with the quantiles of
all residuals pooled across hours, become the spread added to the fitted
linear predictor as `exp(mu + q * 1.2)`.  Blend weight and scale were picked
by leave-one-out on the shipped curves.  When fewer than 8 usable training
curves survive, the program falls back to a persistence forecast around `p1`
with a lognormal spread that grows with sqrt(h - 1).

Standard library only, no network, deterministic.
"""

import glob
import json
import math
import os

# Contest quantile levels, in output order.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Weight given to the residual quantiles pooled across all hours (the rest,
# 0.4, goes to the hour's own residual quantiles).
POOLED_WEIGHT = 0.6

# Multiplier on the residual quantiles before exponentiation.
RESIDUAL_SCALE = 1.2

# Comparable curves with |lift| above this are early-access / staggered
# unlocks, whose hour 1 is not the onset the rest of the day continues from.
LIFT_GATE = 6.0

# A comparable must have at least this many populated release-day hours.
MIN_POPULATED_HOURS = 20

# Below this many usable curves the regression is abandoned for persistence.
MIN_USABLE_CURVES = 8

# Release-day hours are 1..24; forecasts cover 2..24.
LAST_HOUR = 24

# Pre-launch idle level is measured over this many hours before day_start.
IDLE_WINDOW_HOURS = 48

# Log-space spread per sqrt(hour) of the persistence fallback.
PERSISTENCE_SIGMA = 0.15

# Standard-normal quantiles matching LEVELS, used by the fallback.
PERSISTENCE_Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Small ridge added to the normal equations so a degenerate design still
# solves instead of dividing by zero.
RIDGE = 1e-9

OUTPUT_LOW = 1e-9
OUTPUT_HIGH = 1e9


def parse_timestamp(text):
    """Convert an ISO 8601 UTC timestamp to epoch seconds."""
    if text is None:
        return 0.0
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    # datetime.fromisoformat handles the offsets used by the fixtures; the
    # calendar arithmetic is done here to avoid importing datetime at all.
    year = int(value[0:4])
    month = int(value[5:7])
    day = int(value[8:10])
    hour = int(value[11:13])
    minute = int(value[14:16]) if len(value) >= 16 else 0
    second = int(value[17:19]) if len(value) >= 19 else 0
    return _days_from_civil(year, month, day) * 86400.0 + hour * 3600.0 + minute * 60.0 + second


def _days_from_civil(year, month, day):
    """Days since 1970-01-01 for a proleptic Gregorian date."""
    year -= 1 if month <= 2 else 0
    era = (year if year >= 0 else year - 399) // 400
    yoe = year - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def hour_offset(timestamp, day_start):
    """Whole-hour offset of a timestamp from the launch's day_start."""
    return int(math.floor((timestamp - day_start) / 3600.0))


def bucket_means(reads, day_start, low, high):
    """Mean viewers per whole-hour bucket with offset in [low, high)."""
    totals = {}
    counts = {}
    for row in reads or ():
        viewers = row.get("viewers")
        if viewers is None:
            continue
        offset = hour_offset(parse_timestamp(row.get("at") or row.get("from")), day_start)
        if offset < low or offset >= high:
            continue
        totals[offset] = totals.get(offset, 0.0) + float(viewers)
        counts[offset] = counts.get(offset, 0) + 1
    return {offset: totals[offset] / counts[offset] for offset in totals}


def median(values):
    """Median of a list of numbers (0.0 for an empty list)."""
    ordered = sorted(values)
    size = len(ordered)
    if size == 0:
        return 0.0
    middle = size // 2
    if size % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def idle_level(reads, day_start):
    """Pre-launch idle level: median of hourly means over 48h before day_start."""
    buckets = bucket_means(reads, day_start, -IDLE_WINDOW_HOURS, 0)
    if not buckets:
        return 1.0
    return max(1.0, median(list(buckets.values())))


def release_hour_one(reads, day_start):
    """Release-day hour-1 mean from the reads, floored at 1.0."""
    buckets = bucket_means(reads, day_start, 0, 1)
    if not buckets:
        return 1.0
    return max(1.0, buckets[0])


def known_hour_one(known):
    """Hour-1 print from `known`, which may be one object or a list."""
    if isinstance(known, dict):
        rows = [known]
    elif isinstance(known, list):
        rows = known
    else:
        rows = []
    for row in rows:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if viewers is not None:
                return max(1.0, float(viewers))
    return None


def describe_curve(hourly, day_start, known=None):
    """Describe a curve as (p1, base, lift, release viewers by hour, populated)."""
    base = idle_level(hourly, day_start)
    p1 = known_hour_one(known) if known is not None else None
    if p1 is None:
        p1 = release_hour_one(hourly, day_start)
    lift = math.log(p1 / base)
    release = {}
    populated = 0
    for row in hourly or ():
        if not isinstance(row, dict):
            continue
        offset = hour_offset(parse_timestamp(row.get("from") or row.get("at")), day_start)
        if offset < 1 or offset > LAST_HOUR:
            continue
        viewers = row.get("viewers")
        if viewers is None:
            continue
        release[offset] = float(viewers)
        populated += 1
    return p1, base, lift, release, populated


def least_squares(rows, targets):
    """Ordinary least squares via normal equations with partial pivoting.

    Returns the coefficient list, or None when there are no rows.  A small
    ridge keeps a degenerate design solvable and deterministic.
    """
    count = len(rows)
    if count == 0:
        return None
    width = len(rows[0])
    normal = [[0.0] * width for _ in range(width)]
    moment = [0.0] * width
    for row, target in zip(rows, targets):
        for i in range(width):
            moment[i] += row[i] * target
            for j in range(width):
                normal[i][j] += row[i] * row[j]
    for i in range(width):
        normal[i][i] += RIDGE
    return _solve(normal, moment)


def _solve(matrix, vector):
    """Gaussian elimination with partial pivoting; None if still singular."""
    size = len(vector)
    work = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(work[r][column]))
        if abs(work[pivot][column]) < 1e-12:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        lead = work[column][column]
        for r in range(column + 1, size):
            factor = work[r][column] / lead
            if factor == 0.0:
                continue
            for c in range(column, size + 1):
                work[r][c] -= factor * work[column][c]
    result = [0.0] * size
    for r in range(size - 1, -1, -1):
        total = work[r][size]
        for c in range(r + 1, size):
            total -= work[r][c] * result[c]
        result[r] = total / work[r][r]
    return result


def empirical_quantile(ordered, tau):
    """Linear-interpolation quantile at tau*(n-1) of an already sorted list."""
    size = len(ordered)
    if size == 0:
        return 0.0
    if size == 1:
        return ordered[0]
    position = tau * (size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def training_curves(comparables):
    """Usable comparable curves after the populated-hours and lift gates."""
    curves = []
    for comparable in comparables or ():
        if not isinstance(comparable, dict):
            continue
        day_start = parse_timestamp(comparable.get("day_start"))
        p1, base, lift, release, populated = describe_curve(comparable.get("hourly"), day_start)
        if populated < MIN_POPULATED_HOURS:
            continue
        if abs(lift) > LIFT_GATE:
            continue
        curves.append({"p1": p1, "base": base, "lift": lift, "release": release})
    return curves


def fit(curves):
    """Fit one regression per hour; return coefficients, own and pooled quantiles."""
    coefficients = {}
    residuals = {}
    pooled = []
    for hour in range(2, LAST_HOUR + 1):
        rows = []
        targets = []
        for curve in curves:
            viewers = curve["release"].get(hour)
            if viewers is None or viewers <= 0:
                continue
            rows.append([1.0, math.log(curve["p1"]), curve["lift"]])
            targets.append(math.log(viewers))
        solved = least_squares(rows, targets)
        if solved is None:
            continue
        coefficients[hour] = solved
        hour_residuals = []
        for row, target in zip(rows, targets):
            predicted = solved[0] + solved[1] * row[1] + solved[2] * row[2]
            hour_residuals.append(target - predicted)
        hour_residuals.sort()
        residuals[hour] = hour_residuals
        pooled.extend(hour_residuals)
    pooled.sort()
    pooled_quantiles = [empirical_quantile(pooled, tau) for tau in LEVELS]
    own_quantiles = {
        hour: [empirical_quantile(values, tau) for tau in LEVELS]
        for hour, values in residuals.items()
    }
    return coefficients, own_quantiles, pooled_quantiles


def blended_quantiles(own, pooled):
    """Blend an hour's own residual quantiles toward the pooled ones."""
    if own is None:
        return list(pooled)
    return [
        (1.0 - POOLED_WEIGHT) * own[i] + POOLED_WEIGHT * pooled[i]
        for i in range(len(LEVELS))
    ]


def clamp_row(row):
    """Clamp to the allowed range and force non-decreasing order."""
    clamped = [min(OUTPUT_HIGH, max(OUTPUT_LOW, value)) for value in row]
    for i in range(1, len(clamped)):
        if clamped[i] < clamped[i - 1]:
            clamped[i] = clamped[i - 1]
    return clamped


def persistence_forecast(p1):
    """Fallback: p1 held flat, lognormal spread growing with sqrt(h - 1)."""
    rows = []
    for hour in range(2, LAST_HOUR + 1):
        width = PERSISTENCE_SIGMA * math.sqrt(hour - 1)
        quantiles = [p1 * math.exp(z * width) for z in PERSISTENCE_Z]
        rows.append({"hour": hour, "q": clamp_row(quantiles)})
    return rows


def forecast(case):
    """Forecast hours 2..24 for one case."""
    day_start = parse_timestamp(case.get("day_start"))
    history = case.get("history") or []
    known = case.get("known")
    known_p1 = known_hour_one(known)
    base = idle_level(history, day_start)
    p1 = known_p1 if known_p1 is not None else release_hour_one(history, day_start)
    p1 = max(1.0, p1)
    lift = math.log(p1 / base)

    curves = training_curves(case.get("comparables"))
    if len(curves) < MIN_USABLE_CURVES:
        return persistence_forecast(p1)

    coefficients, own_quantiles, pooled_quantiles = fit(curves)
    features = [1.0, math.log(p1), lift]
    kept = persistence_forecast(p1)
    rows = []
    for hour in range(2, LAST_HOUR + 1):
        solved = coefficients.get(hour)
        if solved is None:
            # Degenerate design for this hour: hold the fallback row.
            rows.append({"hour": hour, "q": list(kept[hour - 2]["q"])})
            continue
        mu = solved[0] + solved[1] * features[1] + solved[2] * features[2]
        quantiles = blended_quantiles(own_quantiles.get(hour), pooled_quantiles)
        values = [math.exp(mu + q * RESIDUAL_SCALE) for q in quantiles]
        rows.append({"hour": hour, "q": clamp_row(values)})
    return rows


def main():
    """Read every case, write its forecast, print nothing."""
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(input_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = {"hours": forecast(case)}
        name = os.path.splitext(os.path.basename(path))[0] + ".out"
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    main()
