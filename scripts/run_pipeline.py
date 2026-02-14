import os
import subprocess
import sys

# Get the directory where this script is located (scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of scripts/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Run command from the project root so scripts find data/ correctly
        subprocess.check_call(command, shell=True, cwd=PROJECT_ROOT)
        print(f">>> {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Error during {description}: {e}")
        sys.exit(1)

def main():
    print("===========================================")
    print("   NAGOYA PROJECT - PUBLIC RELEASE PIPELINE")
    print("===========================================")
    
    # 0. Sync CETRAM Data
    run_command("python scripts/import_cetram_data.py", "Step 0: Syncing CETRAM Comparison Data")

    # 1. Calculate Metrics & Complexity (Julia)
    try:
        subprocess.check_call("julia --version", shell=True, stdout=subprocess.DEVNULL)
        run_command("julia scripts/pipeline_step1_calculate_metrics.jl", "Step 1: Calculating Metrics & Complexity (Julia)")
    except subprocess.CalledProcessError:
        print("\n!!! Julia not found or error running Julia. Skipping Step 1.")
        print("    Please ensure Julia is installed and added to PATH.")

    # 2. Generate Figures
    run_command("python scripts/pipeline_step2_fig1.py", "Step 2: Generating Figure 1 (24h Trends)")
    run_command("python scripts/pipeline_step3_fig2.py", "Step 3: Generating Figure 2 (Day/Night Complexity)")
    run_command("python scripts/pipeline_step4_fig3.py", "Step 4: Generating Figure 3 (Circadian Evolution)")
    run_command("python scripts/pipeline_step5_fig4.py", "Step 5: Generating Figure 4 (Correlations)")
    run_command("python scripts/pipeline_step6_supplementary_dashboard.py", "Step 6: Generating Supplementary Dashboard")
    
    print("\n===========================================")
    print("       PIPELINE COMPLETED SUCCESSFULLY       ")
    print("===========================================")

if __name__ == "__main__":
    main()
