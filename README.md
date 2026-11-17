# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, one hour into the release day of Grand Theft Auto VI, how many people
will be watching it live on Twitch in each of the remaining 23 hours of that UTC
day — as five quantiles an hour, so the forecast says both how many and how sure.

## How it works

A launch's audience curve is not a function of the clock hour; it is a function of
how far past its unlock it is. Hour 1's print alone cannot say where on that curve
we stand: a game still waiting on a rolling midnight unlock prints what its idle
channels print, while one that unlocked hours earlier is already falling away from
its peak. The program measures the pre-launch idle level from the case's own
history and uses the ratio of hour 1's print to it as the position on the curve:

    lift = ln(p1 / base),   base = median hourly viewers over the 48 h before the day

It then fits, per hour, the predictive distribution of the log viewer count on
ln(p1) and lift, from the past launch curves the case ships in `comparables`
(38 of them, 33 usable: a launch whose hour 1 is already far past its own unlock
— early access, a staggered PC release, a live-service launch — is left out of
the fit), and takes the five quantiles of each fitted distribution — which is
exactly what a pinball loss scores.

Coefficients are fitted at run time from the case. Nothing is baked, fetched or
installed: standard library only, no network, deterministic.

## Verified

Measured on the four practice cases, not asserted:

- shape: four `.out` files, 2.5 KB each, 23 rows for hours 2..24, five finite
  quantiles each above 0 and non-decreasing;
- determinism: two runs into separate directories are byte identical
  (`cmp` clean), and unchanged by `PYTHONHASHSEED`;
- time: 0.36 s for all four cases, 0.12–0.23 s per case against the 8 s cap;
- fallback: a case with `comparables` emptied still writes a valid forecast,
  holding hour 1's print with a spread that widens as a random walk;
- the practice loss cannot be computed here: the cases ship no answers.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Under the grader it is `python3 /bot/main.py`, one case at a time, reading
`/input/<id>.in` and writing `/output/<id>.out`.

## License

MIT — see [LICENSE](LICENSE).
