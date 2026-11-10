# InverterRL

Reinforcement learning for a home inverter battery under power cuts and time-of-day electricity prices.

A lot of homes in India have an inverter with a battery that just sits there waiting for the next power cut. With a time-of-day tariff the same battery could also save money: charge when power is cheap, run the house from it when power is expensive. The catch is that every unit you use to save money is a unit you don't have when the next cut comes. This project looks at how well different RL methods learn that trade-off, and how quickly they adapt when the power-cut pattern changes with the season.

The problem is small enough (1,056 states, 3 actions) that I could also solve it exactly with value iteration. So every learning method gets compared with the true optimum, not just with the other methods.

## What's in the model

- **State:** battery level (0 to 10 units), hour of the day, and how long the grid has been off (on, 1 h, 2 h, 3+ h).
- **Actions:** charge, hold or discharge. During a cut the action doesn't matter, the inverter runs the house from the battery.
- **Reward:** minus the electricity bill for the hour, minus a penalty κ for every hour of blackout (cut with an empty battery). κ = 50 by default, and I also tried 20 and 100.
- **Power cuts:** estimated from one year (2018) of minute-wise voltage readings at a domestic connection in Guddiangadi, Karnataka, from the ESMI dataset. Cuts start about twice as often in April to June, so there's a "normal" and a "summer" pattern. The chance of the power coming back drops the longer a cut lasts, which is why the state has the hours-off counter.
- **Prices:** solar hours 10:00–17:59 at ₹4.80, evening peak 18:00–21:59 at ₹7.20, everything else ₹6.00. This follows the 2023 time-of-day rule with an assumed normal rate of ₹6, so it's a scenario rather than a real bill.

## Methods

- Value iteration (exact optimum, used as the benchmark)
- A hand-written rule (charge in solar hours, discharge at peak if the battery is above 3)
- Monte Carlo control, SARSA and Q-learning (tabular)
- DQN (small network, replay buffer, target network)

## Results

Everything is averaged over 5 training seeds. Regret is how much worse a policy scores than the optimum, in % of the optimum, measured on the same simulated power cuts.

| | Monte Carlo | SARSA | Q-learning | DQN | Rule |
|---|---|---|---|---|---|
| Final regret, normal pattern (κ = 50) | 5.9% | 0.5% | 0.6% | 2.2% | 1.1% |
| Seeds that reached 2% | 0/5 | 5/5 | 5/5 | 1/5 | – |
| Episodes to reach 2% | – | ~23,000 | ~66,000 | – | – |
| Final regret at κ = 20 / κ = 100 | 5.9% / 8.6% | 0.7% / 1.6% | −0.6% / 1.8% | 1.5% / 34% | 0.6% / 3.2% |

The short version:

- SARSA and Q-learning learn policies within 1% of the optimum and slightly better than the rule. Monte Carlo never gets close, because with γ = 0.99 its returns add up hundreds of hours of random cuts.
- When the pattern switches to summer, the TD methods win back about half of the 5.7% you lose by not adapting within the first 2,000 episodes, but after that they hover around 2–3% of the new optimum. Resetting ε to 0.5 at the switch helps a bit at the start. Keeping ε constant at 0.10 didn't help.
- DQN was unstable. It gets close early on, then its regret jumps around, and after the summer switch (and at κ = 100) it gets much worse. I didn't get to the bottom of why.

## Repository layout

```
inverter-rl/
  env.py               the environment (Gymnasium)
  estimate_cuts.py     works out the cut probabilities from the ESMI files
  dp.py                value iteration
  baseline.py          the hand-written rule
  evaluate.py          runs a fixed policy for N days and returns score, bill, blackout hours
  agents/              Monte Carlo, SARSA, Q-learning, DQN
  run_experiments.py   Experiments 1 to 3 (all methods, penalties, seeds)
  analyze.py           tables and figures from the results
  export_policies.py   exports policies for the simulator
  results/             saved results of the main run (.npz) and summary.json
  test_*.py            tests
simulator/             browser simulator (index.html, sim.js, policies.json)
```

## Running it

I used Python 3.12. From the repo root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy gymnasium torch matplotlib pytest
```

(Versions I had: numpy 2.5, gymnasium 1.4, torch 2.15, matplotlib 3.11, pytest 9.2.)

Everything below is run from inside `inverter-rl/`.

**Tests:**

```
python -m pytest -q
```

**The data** isn't in the repo because the files are over 500 MB. You only need it if you want to re-estimate the cut probabilities; they are already written into `env.py`. Download it from the Harvard Dataverse (Prayas Energy Group, "Minute-wise voltage data collected in Electricity Supply Monitoring Initiative", doi:10.7910/DVN/CLLZZM, CC0 licence) and put these files in a `data/` folder at the repo root:

```
data/ESMI minute-wise voltage data 2018 Jan-May.csv
data/ESMI minute-wise voltage data 2018 June-Aug.csv
data/ESMI minute-wise voltage data 2018 Sep-Dec.csv
data/ESMI location information.csv
```

Then `python estimate_cuts.py` prints the tables.

**Experiments:**

```
python run_experiments.py
```

This runs 60 jobs (4 methods × 3 penalties × 5 seeds) in parallel on all CPU cores. It took about 80 minutes on my 8-core Mac. Results go to `results/`, and finished jobs are skipped if you stop and restart it. The results from my run are already in `results/`, so you can skip this step.

**Tables and figures:**

```
python analyze.py
```

Prints all the numbers, writes `results/summary.json`, and saves the figures to `../report/figures/`. Takes a couple of minutes.

## Simulator

`simulator/` is a small web page that plays the battery hour by hour. You can put the optimal policy, the rule and any of the trained agents side by side on exactly the same power cuts, switch between the normal and summer patterns, and change the penalty. It uses the real trained policies from `results/` (exported with `python export_policies.py`).

It has to be served, not opened as a file, because it loads `policies.json`:

```
cd simulator
python3 -m http.server
```

Then open http://localhost:8000.

## Limitations

- The battery is lossless and doesn't age, demand is fixed at one unit per hour, and there's no solar panel.
- All cut probabilities come from one location in one year, and some are based on only a handful of cuts.
- The tariff is a scenario. Most homes are still billed by monthly slabs.
- Only 5 seeds per setting, so the confidence intervals are wide.
- The seasonal change is one sudden switch, and the "reset ε" setting assumes the agent is told about it.

## References

- Prayas (Energy Group), Minute-wise voltage data collected in Electricity Supply Monitoring Initiative, Harvard Dataverse, 2019. https://doi.org/10.7910/DVN/CLLZZM
- Ministry of Power, Electricity (Rights of Consumers) Amendment Rules, 2023.
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- V. Mnih et al., Human-level control through deep reinforcement learning, *Nature* 518, 2015.
