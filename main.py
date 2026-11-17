"""Forecast GTA VI launch-day Twitch viewers, hour by hour.

Method, in one paragraph: every past big launch in the case's
``comparables`` tells us how its release day unfolded relative to its
first hour.  Twitch viewership on a launch day is not flat: it dips in
the small hours of UTC, jumps when the United States unlocks, then
rises into the US daytime / European evening and falls again late.
The size of that rise depends strongly on how big hour 1 already was:
a launch whose hour 1 is already huge (most of the world watching)
barely grows, while one whose hour 1 is small (only New Zealand,
Australia and the UK in so far) can grow several-fold.  So for each
hour h = 2..24 we fit, across comparable launches, a weighted
least-squares line

    log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)

with b_h negative for the late hours, and take the fitted value at
this case's hour-1 print as the median log ratio.  The spread is the
residual standard deviation of that line (floored at 0.25 log units,
capped at 0.80), and the five quantiles are the median times
exp(z_tau * sd) with z the standard normal quantiles.

Which comparables: only launches whose unlock was a rolling midnight
local launch (``rolling`` or ``midnight local`` in the unlock note),
which is exactly how GTA VI unlocks, with hour 1 at least 10,000
viewers (smaller categories are sampled only about every 12 hours by
SullyGnome, so their hour-1 point is unreliable).  GTA VI's own
pre-release category history is not a launch and is excluded.  If
fewer than eight rolling launches are available, fall back to all
eligible comparables.  The hour-1 print used for the regression is
clipped to the training range so a record-breaking hour 1 does not
extrapolate the line into nonsense.

Standard library only, deterministic, no network, no randomness:
two runs on the same case write the same bytes.  Runs in well under
a second on a 1 MB case file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
MIN_HOUR1 = 10000
MIN_ROLLING = 8
SD_FLOOR = 0.25
SD_CAP = 0.80


def round2(x):
    """Round to 2 decimals, half up, returning an int for whole values."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day(comp):
    """Hour -> viewers for the comparable's release day (hours 1..24)."""
    try:
        day_start = parse_time(comp["day_start"])
    except Exception:
        return {}
    out = {}
    for row in comp.get("hourly", []):
        try:
            t = parse_time(row["from"])
            viewers = row["viewers"]
        except Exception:
            continue
        if viewers is None or viewers <= 0:
            continue
        delta = (t - day_start).total_seconds() / 3600.0
        hour = int(delta) + 1
        if 1 <= hour <= 24 and hour not in out:
            out[hour] = float(viewers)
    return out


def is_launch(comp):
    name = comp.get("name", "")
    if "Grand Theft Auto VI category" in name:
        return False
    unlock = comp.get("unlock", "")
    if unlock.startswith("Not a launch"):
        return False
    return True


def is_rolling(comp):
    unlock = comp.get("unlock", "").lower()
    return "rolling" in unlock or "midnight local" in unlock


def eligible_comparables(case):
    """(rolling_maps, all_maps): release-day maps of eligible launches."""
    rolling = []
    all_maps = []
    for comp in case.get("comparables", []):
        if not is_launch(comp):
            continue
        day = release_day(comp)
        if 1 not in day or day[1] < MIN_HOUR1:
            continue
        # Need a usable release day, not just an hour-1 point.
        present = sum(1 for h in range(2, 25) if h in day)
        if present < 15:
            continue
        all_maps.append(day)
        if is_rolling(comp):
            rolling.append(day)
    return rolling, all_maps


def fit_hour(maps, hour, log_p1):
    """Return (median log ratio, sd) for one hour, by regression on log p1."""
    pts = []
    for day in maps:
        if hour in day and day[hour] > 0:
            pts.append((math.log(day[1]), math.log(day[hour] / day[1])))
    if len(pts) < 4:
        # Too little data: persistence with a wide spread.
        return 0.0, 0.60
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    n = len(pts)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        b = 0.0
    else:
        b = sum((x - mx) * (y - my) for x, y in pts) / den
    a = my - b * mx
    lo, hi = min(xs), max(xs)
    x_pred = min(max(log_p1, lo), hi)
    pred = a + b * x_pred
    resid = [y - (a + b * x) for x, y in pts]
    var = sum(r * r for r in resid) / n
    sd = math.sqrt(var)
    sd = max(SD_FLOOR, min(SD_CAP, sd))
    return pred, sd


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError("%s asks for levels %s, not %s" % (name, case.get("levels"), LEVELS))
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    rolling, all_maps = eligible_comparables(case)
    maps = rolling if len(rolling) >= MIN_ROLLING else all_maps
    if len(maps) < 4:
        # Last resort: the house baseline (persistence, widening spread).
        hours = []
        for row in case["forecast"]:
            h = row["hour"]
            q = [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]
            hours.append({"hour": h, "q": q})
        return {"hours": hours}

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        # If the rolling set is thin for this particular hour (some
        # launches lack that hour's point), fall back to all eligible
        # launches for that hour rather than to persistence.
        n_rolling = sum(1 for day in maps if h in day)
        hour_maps = maps if n_rolling >= 4 else all_maps
        med_log, sd = fit_hour(hour_maps, h, log_p1)
        q = []
        prev = 0.0
        for z in Z:
            v = p1 * math.exp(med_log + z * sd)
            v = round2(v)
            v = max(1.0, min(1_000_000_000.0, float(v)))
            # Keep the row non-decreasing after rounding.
            if v < prev:
                v = prev
            prev = v
            q.append(v)
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
