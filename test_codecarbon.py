"""
Test CodeCarbon to verify it's working properly
"""

import time
import os

print("Testing CodeCarbon...")
print("="*60)

try:
    from codecarbon import EmissionsTracker
    print("[OK] CodeCarbon imported successfully")
    
    # Test basic initialization
    tracker = EmissionsTracker(
        output_dir="./results",
        log_level="warning"
    )
    print("[OK] EmissionsTracker initialized")
    
    # Test starting tracker
    tracker.start()
    print("[OK] Tracker started")
    
    # Do some work (simulate computation)
    print("\nSimulating computation for 5 seconds...")
    time.sleep(5)
    
    # Stop tracker
    tracker.stop()
    print("[OK] Tracker stopped")
    
    # Check if emissions data was collected - read from CSV
    import pandas as pd
    csv_path = os.path.join("./results", "emissions.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                last_row = df.iloc[-1]
                energy = float(last_row.get('energy_consumed', 0))
                emissions = float(last_row.get('emissions', 0))
                duration = float(last_row.get('duration', 0))
                
                print("\n" + "="*60)
                print("CodeCarbon Results (from CSV):")
                print(f"  Energy consumed: {energy:.6f} kWh")
                print(f"  CO2 emitted: {emissions:.9f} g")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  CPU Power: {last_row.get('cpu_power', 'N/A')} W")
                print(f"  GPU Power: {last_row.get('gpu_power', 'N/A')} W")
                print("="*60)
                
                if energy > 0 or emissions > 0:
                    print("\n[SUCCESS] CodeCarbon is tracking emissions correctly!")
                else:
                    print("\n[WARN] CodeCarbon is running but no emissions data collected yet.")
            else:
                print("\n[WARN] CSV file is empty")
        except Exception as e:
            print(f"\n[WARN] Could not read CSV: {e}")
    else:
        print("\n[WARN] No emissions CSV file found yet")
    
    # Check if output file was created
    output_files = [f for f in os.listdir("./results") if "emissions" in f.lower() or ".csv" in f.lower()]
    if output_files:
        print(f"\n[OK] Output files found: {output_files[:3]}")
    else:
        print("\n[INFO] No emissions CSV files found yet (may be created after first full run)")
    
    print("\n" + "="*60)
    print("CodeCarbon test complete!")
    print("="*60)
    
except ImportError as e:
    print(f"[FAIL] CodeCarbon not installed: {e}")
    print("\nInstall with: pip install codecarbon")
except Exception as e:
    print(f"[ERROR] CodeCarbon test failed: {e}")
    import traceback
    traceback.print_exc()

