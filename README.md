# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Fauzii99/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Fauzii99/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour: for each of hours 2 to 24 (UTC, 19 Nov 2026) it writes five quantiles of that hour's concurrent viewers, anchored on hour 1's print and on how past rolling-midnight launches unfolded relative to their first hour.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

`main.py` (Python 3.13, standard library only, no network) reads the case file, builds each comparable launch's release-day curve, and for every hour fits `log(viewers_h / viewers_1) = a + b * log(viewers_1)` across launches with a rolling midnight-local unlock. A launch whose hour 1 is already huge grows little; one whose hour 1 is small can grow several-fold, and the regression captures that. The spread is the residual standard deviation of that fit. On the four practice launches this scores a mean pinball loss of 0.163, against the house baseline's 0.387.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pg1dyky2&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
