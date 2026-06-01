import subprocess
import sys

SCRIPTS = [
    "a1_models_baseline_auto",
    "a1_models_baseline_nonauto",
    "a2_models_hurdle_stage1_logit",
    "a3_models_hurdle_stage2_nb",
    "a4_models_hurdle_combined",
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
