import subprocess
import sys

SCRIPTS = [
    "a0_setup_directories",
    "a1_load_raw_data",
    "a2_data_imputation_auto",
    "a2_data_imputation_nonauto",
    "a3_eda",
    "a4_modeldat_baseline_auto",
    "a4_modeldat_baseline_nonauto",
    "a5_modeldat_hurdle_auto",
    "a5_modeldat_hurdle_nonauto"
]

package = __package__

for script in SCRIPTS:
    module = f"{package}.{script}"
    # flush=True so banners appear before subprocess output when stdout is redirected (non-TTY).
    print(f"\n{'='*60}", flush=True)
    print(f"Running {module}", flush=True)
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run([sys.executable, "-u", "-m", module])
    if result.returncode != 0:
        print(f"\nFAILED: {module} exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)

print(f"\n{'='*60}", flush=True)
print(f"All {package} scripts completed successfully.", flush=True)
print(f"{'='*60}", flush=True)
