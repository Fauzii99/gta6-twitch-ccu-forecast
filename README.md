# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

This program forecasts hourly Twitch concurrent-viewer counts for every hour of a
game launch day, hours 2 through 24, as five quantiles per hour (0.05, 0.25, 0.5,
0.75, 0.95) of a predictive distribution over ln(viewers). Input is a JSON case
holding hour 1's print, the release-day checkpoint, every read taken before it,
and a set of comparable past launches; output is a 23-row `{"hours": [...]}` JSON
file with one five-quantile row per hour, written next to the input as `X.out`.

The method locates the case on its audience curve before it forecasts it. Each
curve is described by a pre-launch idle level (the median of hourly read means
over the 48 hours before `day_start`) and a lift, `ln(p1 / base)`, which says how
far hour 1 stands above idle and therefore how much of the spike has already
happened. Comparables with fewer than 20 populated release-day hours, or a lift
beyond 6.0 (early-access and staggered-unlock launches whose hour 1 is not the
onset the rest of the day continues from), are dropped. What remains trains, at
run time and per hour, an ordinary least squares regression of ln(viewers) on
`[1.0, ln(p1), lift]`; its residuals give the spread, with each hour's own
residual quantiles blended 40/60 toward the quantiles of all residuals pooled
across hours, and the forecast is `exp(mu + q * 1.2)`. The contest scores a
pinball loss on the log scale, and the minimiser of a pinball loss at level tau
is the tau-quantile of the predictive distribution — so quantiles of a
distribution over ln(viewers) are the answer, not point estimates. Fewer than 8
usable comparables triggers a persistence fallback that holds hour 1's print and
widens lognormally with sqrt(h - 1).

## Run

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Defaults are `/input` and `/output`; input files are read in sorted glob order so
two runs write identical bytes. Standard library only, no network.

## License

MIT
