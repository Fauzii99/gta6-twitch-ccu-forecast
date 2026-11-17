"""GTA VI launch-day Twitch concurrent-viewer forecast.

Reads one case file (/input/<id>.in) and writes /output/<id>.out: 23 rows for
hours 2..24 of the UTC release day, five quantiles each (5%, 25%, 50%, 75%, 95%).

Method. The audience curve of a game launch is not a function of the clock hour
but of how far past its unlock it is. Hour 1's print alone cannot say where we
are: a launch still waiting on a rolling midnight unlock has an hour-1 print no
different from its idle channel, while one that unlocked hours earlier is already
falling away from its peak. So the case's own history is used to measure the
pre-launch idle level, and the ratio of hour 1's print to it -- `lift` -- carries
that position:

    lift = ln(p1 / base),  base = median hourly viewers over the 48 h before the day

The predictive distribution of ln(y_h) is then a linear function of ln(p1) and
lift, with an empirical residual quantile attached per hour:

    ln(y_h) ~ c0[h] + c1[h]*ln(p1) + c2[h]*lift + q[h]

`q[h]` is the tau-quantile of the fitted residuals, shrunk toward the pooled
residual distribution and scaled up slightly to widen coverage: the contest's
loss is a pinball loss, whose minimiser at level tau is exactly the tau-quantile
of the predictive distribution, so a calibrated spread is not decoration here --
it is the score.

The coefficients are fitted at run time from the launch curves the case itself
ships in `comparables` (38 past releases, 33 of them usable after the two
exclusions below), and checked by leave-one-out on those curves' own hours 2..24.
Nothing is baked and nothing is fetched: the program is deterministic, offline
and stdlib-only. The practice cases' answers are withheld, so no local run can
measure the contest's loss -- only this leave-one-out proxy on the shipped curves.

Written to run as `python3 /bot/main.py` under the grader's `-P -s` flags.
"""
import json
import math
import os
import statistics
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Residual quantiles are pulled this far toward the pooled distribution, then
# widened by this factor. Both checked by leave-one-out on the shipped curves:
# SCALE 1.2 is the grid's optimum and SHRINK 0.3 its, with 0.6 within 0.4% of it
# (0.452846 against 0.451121 on 15,180 held-out items), so 0.6 is kept.
SHRINK = 0.6
SCALE = 1.2

# A curve whose hour-1 print sits this far above its own pre-launch idle level is
# already hours past its unlock (early access, a staggered PC unlock, a
# live-service launch), so hour 1 is not the onset the rest of the day continues
# from. Those are left out of the fit rather than allowed to drag it. Note the
# gate bounds the training rows, not the target: a case can sit above it (two of
# the four practice cases do, at 6.69 and 6.25), and its lift term is then
# extrapolated. Dropping the gate entirely scores 0.428333 by the same
# leave-one-out against 0.433775 with it, a 1.3% gain too small to act on when
# the practice answers are withheld -- so the gate stays, and this is the risk.
MAX_ABS_LIFT = 6.0

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def hour_of(iso):
    """Hours since the Unix epoch for an ISO 8601 UTC timestamp."""
    d = iso.rstrip("Z")
    day, clock = d.split("T")
    y, mo, da = (int(x) for x in day.split("-"))
    hh = int(clock.split(":")[0])
    yy = y - (1 if mo <= 2 else 0)
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    doy = (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + da - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return (era * 146097 + doe - 719468) * 24 + hh


def base_level(points, day_start, hours=48):
    """Median hourly viewer count over the `hours` before the release day."""
    d0 = hour_of(day_start)
    buckets = {}
    for p in points:
        v = p.get("viewers")
        if v is None:
            continue
        stamp = p.get("at") or p.get("from")
        if not stamp:
            continue
        off = hour_of(stamp) - (d0 - hours)
        if 0 <= off < hours:
            buckets.setdefault(off, []).append(float(v))
    if not buckets:
        return 1.0
    means = [sum(v) / len(v) for v in buckets.values()]
    return max(1.0, statistics.median(means))


def curve_of(points, day_start):
    """Viewer count per release-day hour, 1..24, as hour means."""
    d0 = hour_of(day_start)
    buckets = {}
    for p in points:
        v = p.get("viewers")
        if v is None:
            continue
        stamp = p.get("from") or p.get("at")
        if not stamp:
            continue
        off = hour_of(stamp) - d0 + 1
        if 1 <= off <= 24:
            buckets.setdefault(off, []).append(float(v))
    return {h: sum(v) / len(v) for h, v in buckets.items()}


def describe(points, day_start):
    """p1, pre-launch base and lift for one launch curve."""
    curve = curve_of(points, day_start)
    if len(curve) < 20:
        return None
    p1 = max(1.0, float(curve.get(1) or 0.0))
    base = base_level(points, day_start)
    return {"curve": curve, "p1": p1, "base": base, "lift": math.log(p1 / base)}


def ols(rows, ys):
    """Least squares through the normal equations, tiny ridge for stability."""
    n, k = len(rows), len(rows[0])
    a = [[sum(rows[i][p] * rows[i][q] for i in range(n)) + (1e-9 if p == q else 0.0)
          for q in range(k)] for p in range(k)]
    b = [sum(rows[i][p] * ys[i] for i in range(n)) for p in range(k)]
    m = [a[i] + [b[i]] for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, k + 1):
            m[col][j] /= div
        for r in range(k):
            if r != col and m[r][col]:
                factor = m[r][col]
                for j in range(col, k + 1):
                    m[r][j] -= factor * m[col][j]
    return [m[i][k] for i in range(k)]


def empirical_quantile(values, tau):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def fit_curves(descs):
    """Per-hour coefficients and residual quantiles, shrunk toward the pooled set."""
    coef, own, pooled_res = {}, {}, []
    for h in range(2, 25):
        rows, ys = [], []
        for d in descs:
            y = d["curve"].get(h)
            if not y or y <= 0:
                continue
            rows.append([1.0, math.log(d["p1"]), d["lift"]])
            ys.append(math.log(y))
        if len(rows) < 8:
            coef[h] = None
            continue
        c = ols(rows, ys)
        if c is None:
            coef[h] = None
            continue
        coef[h] = c
        res = [ys[i] - sum(c[j] * rows[i][j] for j in range(3)) for i in range(len(rows))]
        own[h] = [empirical_quantile(res, t) for t in LEVELS]
        pooled_res.extend(res)
    pooled = ([empirical_quantile(pooled_res, t) for t in LEVELS]
              if pooled_res else [0.0] * len(LEVELS))
    model = {}
    for h in range(2, 25):
        if coef.get(h) is None:
            model[h] = None
            continue
        rq = own.get(h) or pooled
        model[h] = (coef[h], [(1.0 - SHRINK) * a + SHRINK * b for a, b in zip(rq, pooled)])
    return model


def forecast(case):
    """The case's 23 hourly forecasts, as 5 quantiles each."""
    target = describe(case.get("history") or [], case["day_start"])

    # Hour 1's print is authoritative when the case carries it; the history is
    # what a case holds before the checkpoint, and it already includes hour 1.
    # The rules call `known` "one row", so accept either one row or a list of
    # them: the shipped cases use a one-element list.
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    p1 = next((max(1.0, float(k.get("viewers") or 0)) for k in known
               if k.get("hour") == 1), None)
    if target is None:
        base = base_level(case.get("history") or [], case["day_start"])
        target = {"curve": {}, "p1": p1 or 1.0, "base": base,
                  "lift": math.log((p1 or 1.0) / base)}
    elif p1 is not None:
        target["p1"] = p1
        target["lift"] = math.log(p1 / target["base"])

    descs = []
    for comp in case.get("comparables") or []:
        d = describe(comp.get("hourly") or [], comp["day_start"])
        if d and abs(d["lift"]) <= MAX_ABS_LIFT:
            descs.append(d)

    model = fit_curves(descs) if len(descs) >= 8 else {}
    lp, lf = math.log(target["p1"]), target["lift"]

    hours = [int(r["hour"]) for r in (case.get("forecast") or [])] or list(range(2, 25))
    out = []
    for h in hours:
        fitted = model.get(h)
        if fitted is None:
            # No comparables to learn from: hold hour 1's print with a spread
            # that widens like a random walk in log space.
            q = [target["p1"] * math.exp(z * 0.15 * math.sqrt(h - 1))
                 for z in (-1.645, -0.674, 0.0, 0.674, 1.645)]
        else:
            c, rq = fitted
            mu = c[0] + c[1] * lp + c[2] * lf
            q = [max(1e-9, math.exp(mu + r * SCALE)) for r in rq]
        q = [max(1e-9, min(1e9, x)) for x in q]
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        out.append({"hour": h, "q": q})
    return {"hours": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
