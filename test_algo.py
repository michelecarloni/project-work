import logging
import pandas as pd
import os
import time
from Problem import Problem
from src.ThiefSolver import ThiefSolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- CONFIGURATION FOR SAVING RESULTS ---
test_dir = "tests"
test_filename = "results.csv"
test_path = os.path.join(test_dir, test_filename)

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

    all_results = []

    for c in configs:
        print(f"\n>>> Running: N={c['N']}, d={c['d']}, alpha={c['a']}, beta={c['b']}")
        p = Problem(num_cities=c['N'], density=c['d'], alpha=c['a'], beta=c['b'])
        
        # 1. Get Baseline
        baseline_cost = p.baseline()
        
        start_time_pre_algo = time.time()
        # 2. Run Optimization with timing
        solver = ThiefSolver(p)
        elapsed_time_pre_algo = round(time.time() - start_time_pre_algo, 2)
        
        print(f"   Time Pre Algo: {elapsed_time_pre_algo:.2f}s")

        start_time_algo = time.time()
        path, total_cost = solver.getSolution()
        elapsed_time_algo = round(time.time() - start_time_algo, 2)

        improvement = ((baseline_cost - total_cost) / baseline_cost) * 100
        status = "SUCCESS" if total_cost < baseline_cost else "FAILURE"

        print(f"   Baseline: {baseline_cost:,.2f}")
        print(f"   Our Algo: {total_cost:,.2f}")
        print(f"   Improvement: {improvement:.2f}% | Status: {status}")
        print(f"   Time Algo: {elapsed_time_algo:.2f}s")

        result_dict = {
            "N": c['N'],
            "density": c['d'],
            "alpha": c['a'],
            "beta": c['b'],
            "Baseline": baseline_cost,
            "Cost": total_cost,
            "Improvement": improvement,
            "Status": status,
            "Time_Pre_Algo": elapsed_time_pre_algo,
            "Time_Algo": elapsed_time_algo,
            "num_elem": len(path),
            "Path": path
        }
        
        save_result_to_csv(result_dict=result_dict, output_dir=test_dir, filename=test_filename)
        all_results.append(result_dict)

    # --- PRINT SUMMARY TABLE ---
    # Define widths for columns
    w_prob = 28
    w_base = 15
    w_algo = 15
    w_imp = 12
    w_stat = 10
    w_time_pre = 12
    w_time_algo = 12
    w_elem = 10
    w_path = 55  # Wider column for path visibility
    
    total_width = w_prob + w_base + w_algo + w_imp + w_stat + w_time_pre + w_time_algo + w_elem + w_path + 10 # spacing

    print("\n" + "="*total_width)
    print("SUMMARY TABLE - PERFORMANCE COMPARISON")
    print("="*total_width)
    
    # Header
    header = (
        f"{'Problem':<{w_prob}} "
        f"{'Baseline':>{w_base}} "
        f"{'Algorithm':>{w_algo}} "
        f"{'Improv.':>{w_imp}} "
        f"{'Status':>{w_stat}} "
        f"{'Time Pre':>{w_time_pre}} "
        f"{'Time Algo':>{w_time_algo}} "
        f"{'Steps':>{w_elem}} "
        f"{'Path (Start ... End)':<{w_path}}"
    )
    print(header)
    print("-" * total_width)
    
    for result in all_results:
        problem_desc = f"N={result['N']}, d={result['density']}, α={result['alpha']}, β={result['beta']}"
        path = result['Path']
        
        # --- INTELLIGENT PATH FORMATTING ---
        # Logic to preserve the LAST element (0, 0.0) even if the path is truncated
        
        last_elem_str = str(path[-1]) # This is always (0, 0.0)
        
        if len(path) > 3:
            # Construct preliminary string: "First, Second"
            start_str = f"{str(path[0])},{str(path[1])}"
            full_str = f"{start_str},...,{last_elem_str}"
            
            # If the combined string is too long for the column, trim the start but KEEP the end
            if len(full_str) > w_path:
                # Calculate how much space is left for the start part
                remaining_space = w_path - len(last_elem_str) - 5 # 5 accounts for "...,"
                if remaining_space > 5:
                    truncated_start = start_str[:remaining_space]
                    path_str = f"{truncated_start}...,{last_elem_str}"
                else:
                    # If very tight, show only dots and end
                    path_str = f"...,{last_elem_str}"
            else:
                path_str = full_str
        else:
            # Short path fits entirely
            path_str = f"{','.join(map(str, path))}"

        row = (
            f"{problem_desc:<{w_prob}} "
            f"{result['Baseline']:>{w_base},.2f} "
            f"{result['Cost']:>{w_algo},.2f} "
            f"{result['Improvement']:>{w_imp}.2f}% "
            f"{result['Status']:>{w_stat}} "
            f"{result['Time_Pre_Algo']:>{w_time_pre}.2f} "
            f"{result['Time_Algo']:>{w_time_algo}.2f} "
            f"{result['num_elem']:>{w_elem}} "
            f"{path_str:<{w_path}}"
        )
        print(row)
    
    print("="*total_width)
    
    if all_results:
        total_improvements = [r['Improvement'] for r in all_results]
        avg_improvement = sum(total_improvements) / len(total_improvements)
        success_count = sum(1 for r in all_results if r['Status'] == 'SUCCESS')
        
        print(f"\nOverall Statistics:")
        print(f"  Average Improvement: {avg_improvement:.2f}%")
        print(f"  Success Rate: {success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)")
    else:
        print("\nNo results to display.")
    print("="*total_width + "\n")

if __name__ == "__main__":
    run_benchmarks()