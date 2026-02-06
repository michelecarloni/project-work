import logging
import pandas as pd
import os
import time
from Problem import Problem
from src.ThiefSolver import ThiefSolver  # Import class directly

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def save_result_to_csv(result_dict, output_dir="tests", filename="results.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, filename)
    df = pd.DataFrame([result_dict])
    if os.path.isfile(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def run_benchmarks():
    configs = [
        {"N": 100,  "d": 0.2, "a": 1, "b": 2},
        {"N": 100,  "d": 1.0, "a": 1, "b": 1},
        {"N": 100,  "d": 1.0, "a": 1, "b": 2},
        {"N": 100,  "d": 1.0, "a": 2, "b": 1},
        {"N": 1000, "d": 0.2, "a": 1, "b": 1},
        {"N": 1000, "d": 0.2, "a": 1, "b": 2},
        {"N": 1000, "d": 0.2, "a": 2, "b": 2},
        {"N": 1000, "d": 1.0, "a": 1, "b": 1},
        {"N": 1000, "d": 1.0, "a": 2, "b": 1},
        {"N": 1000, "d": 1.0, "a": 1, "b": 2},
    ]

    # Remove old results to start fresh
    if os.path.exists("tests/results.csv"):
        os.remove("tests/results.csv")

    for c in configs:
        print(f"\n>>> Running: N={c['N']}, d={c['d']}, alpha={c['a']}, beta={c['b']}")
        p = Problem(num_cities=c['N'], density=c['d'], alpha=c['a'], beta=c['b'])
        
        # 1. Get Baseline
        baseline_cost = p.baseline()
        
        # 2. Run Optimization (Directly via Class to get Cost instantly)
        # This bypasses the slow external check

        solver = ThiefSolver(p)
        path, total_cost = solver.getSolution()
        
        improvement = ((baseline_cost - total_cost) / baseline_cost) * 100
        status = "SUCCESS" if total_cost < baseline_cost else "FAILURE"

        print(f"   Baseline: {baseline_cost:,.2f}")
        print(f"   Our Algo: {total_cost:,.2f}")
        print(f"   Improvement: {improvement:.2f}% | Status: {status}")

        save_result_to_csv({
            "N": c['N'],
            "density": c['d'],
            "alpha": c['a'],
            "beta": c['b'],
            "Baseline": baseline_cost,
            "Cost": total_cost,
            "Improvement": improvement,
            "Status": status
        })

if __name__ == "__main__":
    run_benchmarks()