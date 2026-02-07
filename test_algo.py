import logging
import pandas as pd
import os
import time
from Problem import Problem
from src.ThiefSolver import ThiefSolver  # Import class directly

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

test_path = "tests/results_10_s.csv"

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
        {"N": 100,  "d": 0.2, "a": 1, "b": 1},
        {"N": 100,  "d": 0.2, "a": 2, "b": 1},
        {"N": 100,  "d": 0.2, "a": 1, "b": 2},
        {"N": 100,  "d": 1, "a": 1, "b": 1},
        {"N": 100,  "d": 1, "a": 2, "b": 1},
        {"N": 100,  "d": 1, "a": 1, "b": 2},
        {"N": 1_000,  "d": 0.2, "a": 1, "b": 1},
        {"N": 1_000,  "d": 0.2, "a": 2, "b": 1},
        {"N": 1_000,  "d": 0.2, "a": 1, "b": 2},
        {"N": 1_000,  "d": 1, "a": 1, "b": 1},
        {"N": 1_000,  "d": 1, "a": 2, "b": 1},
        {"N": 1_000,  "d": 1, "a": 1, "b": 2},
    ]

    # Remove old results to start fresh
    if os.path.exists(test_path):
        os.remove(test_path)

    # Collect all results for summary table
    all_results = []

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

        result_dict = {
            "N": c['N'],
            "density": c['d'],
            "alpha": c['a'],
            "beta": c['b'],
            "Baseline": baseline_cost,
            "Cost": total_cost,
            "Improvement": improvement,
            "Status": status
        }
        
        save_result_to_csv(result_dict)
        all_results.append(result_dict)

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE - PERFORMANCE COMPARISON")
    print("="*100)
    print(f"{'Problem':<20} {'Baseline':>15} {'Algorithm':>15} {'Improvement':>12} {'Status':>10}")
    print("-"*100)
    
    for result in all_results:
        problem_desc = f"N={result['N']}, d={result['density']}, α={result['alpha']}, β={result['beta']}"
        print(f"{problem_desc:<20} {result['Baseline']:>15,.2f} {result['Cost']:>15,.2f} {result['Improvement']:>11.2f}% {result['Status']:>10}")
    
    print("="*100)
    
    # Calculate overall statistics
    total_improvements = [r['Improvement'] for r in all_results]
    avg_improvement = sum(total_improvements) / len(total_improvements)
    success_count = sum(1 for r in all_results if r['Status'] == 'SUCCESS')
    
    print(f"\nOverall Statistics:")
    print(f"  Average Improvement: {avg_improvement:.2f}%")
    print(f"  Success Rate: {success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)")
    print("="*100 + "\n")

if __name__ == "__main__":
    run_benchmarks()