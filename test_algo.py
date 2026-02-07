import logging
import pandas as pd
import os
import time
import numpy as np  # Required for the fast cost calculator
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

# --- EXTERNAL COST CALCULATOR (Optimized with NumPy) ---
def calculate_external_cost(path, problem):
    """
    Computes cost using Vectorized NumPy operations.
    This replaces the slow Python loop with instant C-level array math.
    """
    if not path:
        return 0.0

    # 1. Data Preparation
    # Convert path to arrays for fast processing
    # path structure is list of tuples: [(next_node, gold_collected), ...]
    path_data = np.array(path, dtype=object)
    next_nodes = path_data[:, 0].astype(int)
    golds = path_data[:, 1].astype(float)
    
    # Reconstruct the sequence of visited nodes: [0, n1, n2, ..., 0]
    # We prepend 0 because the thief always starts at the base
    visited_nodes = np.insert(next_nodes, 0, 0)
    
    # 2. Vectorized Distance Lookup
    # We use a list comprehension which is the fastest way to extract data from NetworkX
    g = problem.graph
    dists = np.array([g[u][v]['dist'] for u, v in zip(visited_nodes[:-1], visited_nodes[1:])])
    
    # 3. Vectorized Weight Calculation
    # We need to calculate cumulative weight, but it RESETS to 0 every time we hit node 0.
    # We split the path into "trips" (segments ending at 0) and calculate cumsum for each.
    weights = np.zeros(len(dists))
    
    # Find where the trips end (where next_node is 0)
    # We iterate over these segments - this is very fast as there are few trips compared to nodes
    trip_end_indices = np.where(next_nodes == 0)[0]
    start_idx = 0
    
    for end_idx in trip_end_indices:
        # Get gold for this specific trip
        segment_gold = golds[start_idx : end_idx + 1]
        
        # Calculate cumulative weight
        # We start with 0, then add gold[0], then gold[0]+gold[1]...
        # np.cumsum gives [g0, g0+g1...]. We shift it right by 1 to get the weight *before* the step.
        cum_gold = np.cumsum(segment_gold)
        segment_weights = np.roll(cum_gold, 1)
        segment_weights[0] = 0.0 # First step of any trip carries 0 weight
        
        weights[start_idx : end_idx + 1] = segment_weights
        start_idx = end_idx + 1

    # 4. Vectorized Cost Formula
    # Cost = dist + (alpha * weight)^beta * dist^beta
    alpha = problem.alpha
    beta = problem.beta
    
    # Pre-calculate powers for the whole array at once
    dist_beta = dists ** beta
    weight_term = (alpha * weights) ** beta
    
    # Final formula applied to all steps simultaneously
    step_costs = dists + (weight_term * dist_beta)
    
    return np.sum(step_costs)

def run_benchmarks():
    configs = [
        # {"N": 100,  "d": 0.2, "a": 1, "b": 1},
        # Uncomment for full run
        # {"N": 100,  "d": 0.2, "a": 2, "b": 1},
        # {"N": 100,  "d": 0.2, "a": 1, "b": 2},
        # {"N": 100,  "d": 1, "a": 1, "b": 1},
        # {"N": 100,  "d": 1, "a": 2, "b": 1},
        # {"N": 100,  "d": 1, "a": 1, "b": 2},
        {"N": 1_000,  "d": 0.2, "a": 1, "b": 1},
        {"N": 1_000,  "d": 1, "a": 1, "b": 1},
        # {"N": 1_000,  "d": 0.2, "a": 2, "b": 1},
        # {"N": 1_000,  "d": 0.2, "a": 1, "b": 2},
        # {"N": 1_000,  "d": 1, "a": 2, "b": 1},
        # {"N": 1_000,  "d": 1, "a": 1, "b": 2},
    ]

    if os.path.exists(test_path):
        os.remove(test_path)

    all_results = []

    for c in configs:
        print(f"\n>>> Running: N={c['N']}, d={c['d']}, alpha={c['a']}, beta={c['b']}")
        p = Problem(num_cities=c['N'], density=c['d'], alpha=c['a'], beta=c['b'])
        
        baseline_cost = p.baseline()
        
        start_time_pre_algo = time.time()
        solver = ThiefSolver(p)
        elapsed_time_pre_algo = round(time.time() - start_time_pre_algo, 2)
        
        print(f"   Time Pre Algo: {elapsed_time_pre_algo:.2f}s")

        start_time_algo = time.time()
        path = solver.getSolution()
        elapsed_time_algo = round(time.time() - start_time_algo, 2)

        # FAST COST CALCULATION
        total_cost = calculate_external_cost(path, p)

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