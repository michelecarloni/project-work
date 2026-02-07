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

# --- EXTERNAL COST CALCULATOR (The "Judge") ---
def calculate_external_cost(path, problem):
    """
    Computes cost based on the returned physical path.
    Does not run inside the solver time.
    """
    total_cost = 0.0
    current_weight = 0.0
    current_node = 0
    alpha = problem.alpha
    beta = problem.beta
    
    for next_node, gold_collected in path:
        dist = problem.graph[current_node][next_node]['dist']
        d_beta = dist ** beta
        
        if current_weight == 0:
            step_cost = dist
        else:
            step_cost = dist + ((alpha * current_weight) ** beta) * d_beta
            
        total_cost += step_cost
        current_weight += gold_collected
        
        if next_node == 0:
            current_weight = 0.0
        current_node = next_node
        
    return total_cost

def run_benchmarks():

    configs = [
        # {"N": 100,  "d": 0.2, "a": 1, "b": 1},
        # {"N": 100,  "d": 0.2, "a": 2, "b": 1},
        # {"N": 100,  "d": 0.2, "a": 1, "b": 2},
        # {"N": 100,  "d": 1, "a": 1, "b": 1},
        # {"N": 100,  "d": 1, "a": 2, "b": 1},
        # {"N": 100,  "d": 1, "a": 1, "b": 2},
        {"N": 1_000,  "d": 0.2, "a": 1, "b": 1},
        {"N": 1_000,  "d": 1, "a": 1, "b": 1},
        {"N": 1_000,  "d": 0.2, "a": 2, "b": 1},
        {"N": 1_000,  "d": 0.2, "a": 1, "b": 2},
        {"N": 1_000,  "d": 1, "a": 2, "b": 1},
        {"N": 1_000,  "d": 1, "a": 1, "b": 2},
    ]

    if os.path.exists(test_path):
        os.remove(test_path)

    all_results = []

    for c in configs:
        print(f"\n>>> Running: N={c['N']}, d={c['d']}, alpha={c['a']}, beta={c['b']}")
        p = Problem(num_cities=c['N'], density=c['d'], alpha=c['a'], beta=c['b'])
        
        baseline_cost = p.baseline()
        
        # 1. Pre-Processing Timer
        start_time_pre_algo = time.time()
        solver = ThiefSolver(p)
        elapsed_time_pre_algo = round(time.time() - start_time_pre_algo, 2)
        print(f"   Time Pre Algo: {elapsed_time_pre_algo:.2f}s")

        # 2. Algorithm Timer (Pure Execution)
        start_time_algo = time.time()
        path = solver.getSolution() # Now returns ONLY the path
        elapsed_time_algo = round(time.time() - start_time_algo, 2)

        print(f"START COMPUTE EXTERNAL COST")

        # 3. Post-Processing (Cost Calculation) - OFF THE CLOCK
        total_cost = calculate_external_cost(path, p)

        print(f"END COMPUTE EXTERNAL COST")

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
    w_prob, w_base, w_algo, w_imp = 28, 15, 15, 12
    w_stat, w_time_pre, w_time_algo = 10, 12, 12
    w_elem, w_path = 10, 55
    
    total_width = w_prob + w_base + w_algo + w_imp + w_stat + w_time_pre + w_time_algo + w_elem + w_path + 10

    print("\n" + "="*total_width)
    print("SUMMARY TABLE - PERFORMANCE COMPARISON")
    print("="*total_width)
    
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
        
        last_elem_str = str(path[-1])
        if len(path) > 3:
            start_str = f"{str(path[0])},{str(path[1])}"
            full_str = f"{start_str},...,{last_elem_str}"
            if len(full_str) > w_path:
                remaining_space = w_path - len(last_elem_str) - 5
                if remaining_space > 5:
                    path_str = f"{start_str[:remaining_space]}...,{last_elem_str}"
                else:
                    path_str = f"...,{last_elem_str}"
            else:
                path_str = full_str
        else:
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
        success_count = sum(1 for r in all_results if r['Status'] == 'SUCCESS')
        print(f"\nOverall Statistics:")
        print(f"  Success Rate: {success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)")
    else:
        print("\nNo results to display.")
    print("="*total_width + "\n")

if __name__ == "__main__":
    run_benchmarks()