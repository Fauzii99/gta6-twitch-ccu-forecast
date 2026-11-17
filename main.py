"""GTA VI launch-day Twitch CCU forecast: per-hour quantile regression on the
curve's own shape, coefficients frozen from 24 console single-player launches.

Reads one case at /input/<id>.in, writes /output/<id>.out: 23 rows, hours 2..24,
five quantiles each (5%, 25%, 50%, 75%, 95%), non-decreasing, > 0.

The predictive distribution of ln(y_h) is a linear function of the case's own
features plus an empirical residual quantile per hour:

    ln(y_h) ~ c0[h] + c1[h]*ln(p1) + c2[h]*lift + q[h]

where p1 is hour 1's print and lift = ln(p1 / base), base being the median hourly
viewer count over the 48 h before the release day -- the pre-launch idle level.
lift separates a launch that has already unlocked (high lift, audience falling
away from its first-hour peak) from one still waiting (low lift, audience ahead).

Deterministic: no clock, no randomness, no network, stdlib only.
"""
import json
import math
import os
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# (c0, c1, c2) per hour 2..24, then the five residual quantiles for that hour.
COEF = {
    2:  (-0.594237, 1.044700, -0.020537, -0.409267, -0.171479, -0.004276, 0.153695, 0.447287),
    3:  (-1.362374, 1.099450, -0.047569, -0.589076, -0.211668, 0.017304, 0.236750, 0.532926),
    4:  (-0.664300, 1.025834, -0.025208, -0.598350, -0.257667, 0.045272, 0.268667, 0.561211),
    5:  (-0.298912, 0.984645, 0.038367, -0.688806, -0.253067, -0.009743, 0.289353, 0.686705),
    6:  (0.143248, 0.952272, 0.054637, -0.884818, -0.345217, 0.031629, 0.288905, 0.873218),
    7:  (1.120981, 0.869610, 0.050984, -0.856421, -0.382382, -0.005995, 0.288556, 1.033899),
    8:  (2.142067, 0.783691, 0.056807, -0.828571, -0.270135, -0.052235, 0.275893, 1.009723),
    9:  (2.593287, 0.741792, 0.065281, -0.737742, -0.309051, -0.071388, 0.304263, 1.024565),
    10: (2.698278, 0.732946, 0.069723, -0.670951, -0.371943, -0.087456, 0.271180, 1.024895),
    11: (3.095638, 0.701705, 0.069383, -0.721214, -0.336943, -0.058579, 0.271195, 0.995960),
    12: (3.411736, 0.680906, 0.069693, -0.740294, -0.327214, -0.069599, 0.278910, 0.954478),
    13: (4.021396, 0.641880, 0.068650, -0.704102, -0.345590, -0.058871, 0.236797, 0.905327),
    14: (4.337034, 0.625016, 0.071806, -0.779744, -0.373861, -0.060907, 0.267492, 0.880060),
    15: (5.006127, 0.575785, 0.079759, -0.829211, -0.343383, -0.068452, 0.298983, 0.828933),
    16: (5.376949, 0.551334, 0.077273, -0.750975, -0.341044, -0.083946, 0.288604, 0.799554),
    17: (4.929297, 0.598638, 0.071863, -0.796942, -0.306239, -0.089980, 0.317202, 0.736896),
    18: (5.188201, 0.579764, 0.076097, -0.773916, -0.335895, -0.053650, 0.318722, 0.751014),
    19: (5.515461, 0.555277, 0.083957, -0.737131, -0.341689, -0.089952, 0.318799, 0.741785),
    20: (5.763916, 0.534050, 0.090785, -0.710999, -0.365817, -0.087262, 0.324270, 0.757299),
    21: (5.786013, 0.530292, 0.091091, -0.688371, -0.406286, -0.061927, 0.330785, 0.822890),
    22: (5.719397, 0.535701, 0.084242, -0.737269, -0.338324, -0.034225, 0.342047, 0.765307),
    23: (5.164271, 0.581452, 0.077128, -0.777856, -0.285284, -0.089115, 0.351865, 0.764309),
    24: (4.552063, 0.621098, 0.092024, -0.673820, -0.338565, -0.076996, 0.334269, 0.803228),
}

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


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 1.0
    mid = n // 2
    return float(s[mid]) if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def prelaunch_base(case):
    """Median hourly viewer count over the 48 h before the release day."""
    d0 = hour_of(case["day_start"])
    buckets = {}
    for p in case.get("history") or []:
        v = p.get("viewers")
        if v is None:
            continue
        off = hour_of(p["at"] if "at" in p else p["from"]) - (d0 - 48)
        if 0 <= off < 48:
            buckets.setdefault(off, []).append(float(v))
    means = [sum(v) / len(v) for v in buckets.values()]
    return max(1.0, median(means)) if means else 1.0


def first_print(case):
    for k in case.get("known") or []:
        if k.get("hour") == 1:
            return max(1.0, float(k.get("viewers") or 0))
    return 1.0


def hour_quantiles(p1, lift, h):
    row = COEF[h]
    mu = row[0] + row[1] * math.log(p1) + row[2] * lift
    return [max(1e-9, math.exp(mu + q)) for q in row[3:]]


def forecast_case(case):
    p1 = first_print(case)
    base = prelaunch_base(case)
    lift = math.log(p1 / base)
    rows = []
    for row in case.get("forecast") or [{"hour": h} for h in range(2, 25)]:
        h = int(row["hour"])
        q = hour_quantiles(p1, lift, h)
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
