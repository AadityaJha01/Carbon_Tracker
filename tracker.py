"""
Carbon tracking wrapper using CodeCarbon
"""

import os
from typing import Optional, Dict
import time

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not installed. Carbon tracking will be disabled.")
    print("Install with: pip install codecarbon")


class CarbonTracker:
    """
    Wrapper around CodeCarbon EmissionsTracker for easy integration.
    Falls back to basic time tracking if CodeCarbon is not available.
    """
    
    def __init__(
        self,
        output_dir: str = "./results",
        region: Optional[str] = None,
        log_level: str = "warning"
    ):
        """
        Initialize carbon tracker.
        
        Args:
            output_dir: Directory to save emissions data
            region: ISO 3166-1 alpha-2 country code (e.g., 'IN-TN' for Chennai)
            log_level: Logging level for CodeCarbon
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.region = region
        self.is_tracking = False
        self.start_time = None
        
        if CODECARBON_AVAILABLE:
            try:
                # CodeCarbon initialization - check which parameter name to use
                if region:
                    # Try with country_iso_code (older versions)
                    try:
                        self.tracker = EmissionsTracker(
                            output_dir=output_dir,
                            country_iso_code=region,
                            log_level=log_level
                        )
                    except TypeError:
                        # Fallback to country_code (newer versions)
                        self.tracker = EmissionsTracker(
                            output_dir=output_dir,
                            country_code=region,
                            log_level=log_level
                        )
                else:
                    self.tracker = EmissionsTracker(
                        output_dir=output_dir,
                        log_level=log_level
                    )
                self.use_codecarbon = True
            except Exception as e:
                print(f"Warning: Could not initialize CodeCarbon: {e}")
                self.use_codecarbon = False
        else:
            self.use_codecarbon = False
    
    def start(self):
        """Start tracking emissions"""
        if not self.is_tracking:
            if self.use_codecarbon:
                self.tracker.start()
            else:
                self.start_time = time.time()
            self.is_tracking = True
    
    def stop(self) -> Dict:
        """
        Stop tracking and return emissions data.
        
        Returns:
            Dictionary with energy and emissions metrics
        """
        if self.is_tracking:
            if self.use_codecarbon:
                self.tracker.stop()
                # Get emissions data - CodeCarbon stores data differently
                # Read from CSV file (most reliable method)
                import pandas as pd
                csv_path = os.path.join(self.tracker.output_dir, "emissions.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            # Get the most recent run
                            last_row = df.iloc[-1]
                            emissions_data = {
                                'energy_consumed_kwh': float(last_row.get('energy_consumed', 0)),
                                'emissions_gCO2eq': float(last_row.get('emissions', 0)),
                                'duration_seconds': float(last_row.get('duration', 0)),
                                'cpu_power_watts': float(last_row.get('cpu_power', 0)) if pd.notna(last_row.get('cpu_power')) else None,
                                'gpu_power_watts': float(last_row.get('gpu_power', 0)) if pd.notna(last_row.get('gpu_power')) else None,
                                'ram_power_watts': float(last_row.get('ram_power', 0)) if pd.notna(last_row.get('ram_power')) else None,
                            }
                        else:
                            raise Exception("Empty CSV")
                    except Exception as e:
                        # Fallback if CSV read fails
                        duration = time.time() - self.start_time if self.start_time else 0
                        emissions_data = {
                            'energy_consumed_kwh': 0.0,
                            'emissions_gCO2eq': 0.0,
                            'duration_seconds': duration,
                            'cpu_power_watts': None,
                            'gpu_power_watts': None,
                            'ram_power_watts': None,
                        }
                else:
                    # No CSV file yet, use fallback
                    duration = time.time() - self.start_time if self.start_time else 0
                    emissions_data = {
                        'energy_consumed_kwh': 0.0,
                        'emissions_gCO2eq': 0.0,
                        'duration_seconds': duration,
                        'cpu_power_watts': None,
                        'gpu_power_watts': None,
                        'ram_power_watts': None,
                    }
            else:
                # Fallback: just track time
                duration = time.time() - self.start_time if self.start_time else 0
                emissions_data = {
                    'energy_consumed_kwh': 0.0,
                    'emissions_gCO2eq': 0.0,
                    'duration_seconds': duration,
                    'cpu_power_watts': None,
                    'gpu_power_watts': None,
                    'ram_power_watts': None,
                }
            
            self.is_tracking = False
            return emissions_data
        
        return {}
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

