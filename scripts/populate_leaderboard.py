"""
Populate sample leaderboard entries for local testing.

Creates or appends to `results/leaderboard.csv` with a few realistic runs so the web UI
and recommendation endpoints have data to work with.
"""
import os
import pandas as pd
from datetime import datetime


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'leaderboard.csv')

    rows = [
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'resnet18',
            'accuracy': 85.3,
            'energy_kwh': 0.48,
            'co2_g': 120.5,
            'training_time_sec': 3600,
            'epochs': 20,
            'batch_size': 64,
            'fp16': False,
            'early_stop': False
        },
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'mobilenet_v2',
            'accuracy': 82.1,
            'energy_kwh': 0.22,
            'co2_g': 55.2,
            'training_time_sec': 1800,
            'epochs': 15,
            'batch_size': 128,
            'fp16': True,
            'early_stop': True
        },
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'cnn',
            'accuracy': 78.5,
            'energy_kwh': 0.12,
            'co2_g': 30.8,
            'training_time_sec': 900,
            'epochs': 10,
            'batch_size': 256,
            'fp16': False,
            'early_stop': True
        }
    ]

    df_new = pd.DataFrame(rows)
    # Compute derived metrics
    df_new['accuracy_per_kwh'] = df_new.apply(lambda r: r['accuracy'] / r['energy_kwh'] if r['energy_kwh'] > 0 else 0, axis=1)
    df_new['accuracy_per_co2'] = df_new.apply(lambda r: r['accuracy'] / r['co2_g'] if r['co2_g'] > 0 else 0, axis=1)
    df_new['accuracy_per_hour'] = df_new.apply(lambda r: r['accuracy'] / (r['training_time_sec'] / 3600) if r['training_time_sec'] > 0 else 0, axis=1)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df_new)} sample runs to {csv_path}")


if __name__ == '__main__':
    main()
