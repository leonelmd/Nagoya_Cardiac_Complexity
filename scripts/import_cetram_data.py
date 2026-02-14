
import os
import shutil
import pandas as pd

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Nagoya/public_release
CETRAM_ROOT = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "CETRAM/public_release")
COMPARISON_DIR = os.path.join(ROOT, "data/cetram_comparison")

# Output files
COMPLEXITY_DEST = os.path.join(COMPARISON_DIR, "Complexity_Index_sample.csv")
HRV_DEST = os.path.join(COMPARISON_DIR, "HRV_metrics_cleaned.csv")

def import_cetram_data():
    print(">>> Syncing CETRAM data for comparison...")
    
    # 1. Complexity Index (Sample Entropy)
    # Source: CETRAM/public_release/results/entropy/sample/Complexity_Index_sample.csv
    cetram_comp_path = os.path.join(CETRAM_ROOT, "results/entropy/sample/Complexity_Index_sample.csv")
    
    if os.path.exists(cetram_comp_path):
        print(f"  Found CETRAM Complexity Index at: {cetram_comp_path}")
        shutil.copy2(cetram_comp_path, COMPLEXITY_DEST)
        print("  Updated local copy.")
    else:
        print(f"  WARNING: CETRAM Complexity Index NOT found at {cetram_comp_path}")
        
    # 2. HRV Metrics
    # Source: CETRAM/public_release/results/metrics/HRV_metrics_cleaned.csv
    # Note: Nagoya expects "HRV_metrics_cleaned.csv".
    
    cetram_hrv_path = os.path.join(CETRAM_ROOT, "results/metrics/HRV_metrics_cleaned.csv")
    
    if os.path.exists(cetram_hrv_path):
        print(f"  Found CETRAM HRV Metrics at: {cetram_hrv_path}")
        shutil.copy2(cetram_hrv_path, HRV_DEST)
        print("  Updated local copy.")
    else:
        print(f"  WARNING: CETRAM HRV Metrics NOT found at {cetram_hrv_path}")

    print(">>> CETRAM data sync complete.")

if __name__ == "__main__":
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    import_cetram_data()
