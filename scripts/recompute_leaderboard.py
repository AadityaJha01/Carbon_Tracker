"""Recompute derived leaderboard metrics for existing `results/leaderboard.csv`.

This script recalculates `accuracy_per_kwh`, `accuracy_per_co2`, and `accuracy_per_hour`
using robust guards against zero values.
"""
import os
import pandas as pd


def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'leaderboard.csv')
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        print('No leaderboard.csv found at', csv_path)
        return

    df = pd.read_csv(csv_path)
    EPS = 1e-6

    def safe_div(a, b):
        try:
            a = float(a)
            b = float(b)
        except Exception:
            return None
        if b <= EPS:
            return None
        return a / b

    df['accuracy_per_kwh'] = df.apply(lambda r: safe_div(r.get('accuracy', 0), r.get('energy_kwh', 0)), axis=1)
    df['accuracy_per_co2'] = df.apply(lambda r: safe_div(r.get('accuracy', 0), r.get('co2_g', 0)), axis=1)
    df['accuracy_per_hour'] = df.apply(lambda r: safe_div(r.get('accuracy', 0), (r.get('training_time_sec', 0) / 3600.0) if r.get('training_time_sec', 0) else 0), axis=1)

    backup = csv_path + '.bak'
    print('Backing up', csv_path, 'to', backup)
    os.replace(csv_path, backup)

    df.to_csv(csv_path, index=False)
    print('Recomputed metrics and saved to', csv_path)


if __name__ == '__main__':
    main()
