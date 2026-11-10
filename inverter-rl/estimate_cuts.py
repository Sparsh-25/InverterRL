"""Estimate the power-cut tables in env.py from the ESMI 2018 data.

Data: Prayas (Energy Group), Minute-wise voltage data collected in Electricity
Supply Monitoring Initiative, Harvard Dataverse, doi:10.7910/DVN/CLLZZM (CC0).
Run from inverter-rl/:  python estimate_cuts.py
"""

import csv
import glob
from collections import defaultdict
from datetime import datetime, timedelta

from env import MAX_OFF, period

LOCATION = "Guddiangadi"
FILES = sorted(glob.glob("../data/ESMI minute-wise voltage data 2018 *.csv"))
NO_SUPPLY_VOLTS = 130  # ESMI's own rule
SUMMER_MONTHS = (4, 5, 6)
HOUR = timedelta(hours=1)


def load_hours():
    """{start of hour: True if off}. An hour is off if at least 30 readings are below 130 V.
    Hours with fewer than 45 readings are skipped (monitor offline)."""
    hours = {}
    for path in FILES:
        with open(path, encoding="utf-8-sig") as f:
            rows = csv.reader(f)
            next(rows)  # header
            for row in rows:
                if row[0] != LOCATION:
                    continue
                volts = [float(v) for v in row[3:] if v.strip()]
                if len(volts) < 45:
                    continue
                t = datetime.strptime(row[1], "%d-%m-%Y") + timedelta(hours=int(row[2]))
                hours[t] = sum(v < NO_SUPPLY_VOLTS for v in volts) >= 30
    return hours


def estimate(hours, summer):
    cuts = defaultdict(lambda: [0, 0])  # period -> [cuts started, on-hours]
    restores = defaultdict(lambda: [0, 0])  # hours off so far -> [restored, off-hours]
    off = 0
    for t in sorted(hours):
        if t - HOUR not in hours:  # after a gap, start counting again
            off = 0
        off = min(off + 1, MAX_OFF) if hours[t] else 0
        next_hour = hours.get(t + HOUR)
        if next_hour is None or (t.month in SUMMER_MONTHS) != summer:
            continue
        if off == 0:
            cuts[period(t.hour)][0] += next_hour
            cuts[period(t.hour)][1] += 1
        else:
            restores[off][0] += not next_hour
            restores[off][1] += 1
    return cuts, restores


if __name__ == "__main__":
    hours = load_hours()
    print(f"{LOCATION}: {len(hours)} valid hours of 8,760 in 2018")
    for name, summer in (("normal", False), ("summer", True)):
        cuts, restores = estimate(hours, summer)
        print(f"\n{name}")
        for p in ("solar", "normal", "peak"):
            n_cuts, n_on = cuts[p]
            print(f"  cut   {p:6s} {n_cuts / n_on:.3f}  ({n_cuts} cuts in {n_on} on-hours)")
        for k in range(1, MAX_OFF + 1):
            n_back, n_off = restores[k]
            print(f"  restore after {k}{'+' if k == MAX_OFF else ''}h {n_back / n_off:.3f}  ({n_back} of {n_off})")
