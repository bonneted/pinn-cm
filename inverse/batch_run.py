import sys
import subprocess

noise_strats = ["diff", "exponential", "threshold"]
num_runs = 10

for noise_strat in noise_strats:
    for run in range(num_runs):
        try:
            print(f"Running with noise_strat={noise_strat}, run={run}")
            subprocess.check_call([sys.executable, "elastic_plate.py", noise_strat])
        except subprocess.CalledProcessError as e:
            print(f"Run with noise_strat={noise_strat}, run={run} failed!")
            print(e)
            sys.exit(1)