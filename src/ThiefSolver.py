import networkx as nx
import numpy as np
import random
import copy
import os
import csv

# =========================================================
# STANDALONE EVALUATOR (Fastest Python option)
# =========================================================
def fast_evaluate(genome, dist_matrix, beta_matrix, gold_map, alpha, beta):
    total_cost = 0.0
    current_node = 0
    current_weight = 0.0
    trip_cost = 0.0
    
    for next_city in genome:
        gold_at_next = gold_map[next_city]
        
        # Fast list access
        d_next = dist_matrix[current_node][next_city]
        b_next = beta_matrix[current_node][next_city]
        
        # INLINED COST CALCULATION
        if current_weight == 0:
            cost_to_next = d_next
        else:
            cost_to_next = d_next + ((alpha * current_weight) ** beta) * b_next

        new_weight = current_weight + gold_at_next
        
        d_ret_aft = dist_matrix[next_city][0]
        b_ret_aft = beta_matrix[next_city][0]
        
        d_ret_now = dist_matrix[current_node][0]
        b_ret_now = beta_matrix[current_node][0]
        
        if new_weight == 0:
            cost_ret_after = d_ret_aft
        else:
            cost_ret_after = d_ret_aft + ((alpha * new_weight) ** beta) * b_ret_aft
            
        if current_weight == 0:
            cost_ret_now = d_ret_now
        else:
            cost_ret_now = d_ret_now + ((alpha * current_weight) ** beta) * b_ret_now
            
        cost_out_empty = dist_matrix[0][next_city]
        
        if gold_at_next == 0:
            cost_ret_new = d_ret_aft
        else:
            cost_ret_new = d_ret_aft + ((alpha * gold_at_next) ** beta) * b_ret_aft

        scenario_extend = trip_cost + cost_to_next + cost_ret_after
        scenario_split = trip_cost + cost_ret_now + cost_out_empty + cost_ret_new

        if scenario_extend <= scenario_split:
            trip_cost += cost_to_next
            current_weight = new_weight
            current_node = next_city
        else:
            total_cost += (trip_cost + cost_ret_now)
            current_node = next_city
            current_weight = gold_at_next
            trip_cost = cost_out_empty

    # Final return
    d_final = dist_matrix[current_node][0]
    b_final = beta_matrix[current_node][0]
    
    if current_weight == 0:
        total_cost += (trip_cost + d_final)
    else:
        total_cost += (trip_cost + d_final + ((alpha * current_weight) ** beta) * b_final)
        
    return total_cost, []

class ThiefSolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.num_nodes = len(self.graph.nodes)
        self.density = nx.density(self.graph)
        self.alpha = self.problem.alpha
        self.beta = self.problem.beta
        
        # 1. OPTIMIZATION: Gold Map List
        self.gold_map_list = [0.0] * self.num_nodes
        for i in range(self.num_nodes):
            self.gold_map_list[i] = self.graph.nodes[i]['gold']

        # 2. OPTIMIZATION: Path Cache
        self.path_cache = {} 

        print("Initializing Matrices & Caching Paths (Standard Dijkstra)...")
        d_mat_np, b_mat_np = self._calculate_complex_matrices_and_cache()
        
        self.dist_matrix = d_mat_np.tolist()
        self.beta_matrix = b_mat_np.tolist()
        
        # PARAMETER TUNING
        self.pop_size = 200 if self.num_nodes <= 100 else 150 
        self.generations = 1000 if self.num_nodes <= 100 else 800
        self.mutation_rate = 0.2
        self.elitism_size = 2
        self.tournament_size = 5

    def _calculate_complex_matrices_and_cache(self):
        """
        Uses Standard Dijkstra to calculate matrices AND cache paths.
        Reliable and robust for sparse graphs.
        """
        d_mat = np.zeros((self.num_nodes, self.num_nodes))
        b_mat = np.zeros((self.num_nodes, self.num_nodes))
        
        # This generator yields (source, {target: [path_nodes]})
        path_gen = nx.all_pairs_dijkstra_path(self.graph, weight='dist')

        for source, paths_dict in path_gen:
            # SAVE THE CACHE!
            self.path_cache[source] = paths_dict
            
            for target, path in paths_dict.items():
                if source == target: continue
                l_sum, b_sum = 0.0, 0.0
                for i in range(len(path) - 1):
                    d = self.graph[path[i]][path[i+1]]['dist']
                    l_sum += d
                    b_sum += d ** self.beta
                d_mat[source][target] = l_sum
                b_mat[source][target] = b_sum
                
        return d_mat, b_mat

    def getSolution(self):
        cities = list(range(1, self.num_nodes))
        population = [random.sample(cities, len(cities)) for _ in range(self.pop_size)]
        
        best_fitness = float('inf')
        best_genome = None
        
        generations_without_improvement = 0
        current_mutation_rate = self.mutation_rate

        # Local variables for speed
        pop_size = self.pop_size
        elitism_size = self.elitism_size
        tournament_size = self.tournament_size
        dist_matrix = self.dist_matrix
        beta_matrix = self.beta_matrix
        gold_map = self.gold_map_list
        alpha = self.alpha
        beta = self.beta

        # Prepare CSV Writer
        # N=1000, d=0.2, a=1, b=2 -> "tests/prob_0.2_1_2.csv"
        csv_filename = f"prob_{round(self.density, 1)}_{round(self.alpha, 1)}_{round(self.beta, 1)}.csv"
        csv_dir = "tests"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_path = os.path.join(csv_dir, csv_filename)
        
        # Open file and write header
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Best_Cost", "Avg_Cost", "Stagnation_Count"])

        for gen in range(self.generations):
            scored_pop = []
            sum_fitness = 0.0
            
            for ind in population:
                cost, _ = fast_evaluate(ind, dist_matrix, beta_matrix, gold_map, alpha, beta)
                scored_pop.append((cost, ind))
                sum_fitness += cost
            
            scored_pop.sort(key=lambda x: x[0])
            avg_fitness = sum_fitness / pop_size
            
            # Elitism & Improvement Check
            if scored_pop[0][0] < best_fitness:
                best_fitness = scored_pop[0][0]
                best_genome = copy.deepcopy(scored_pop[0][1])
                generations_without_improvement = 0
                current_mutation_rate = self.mutation_rate
            else:
                generations_without_improvement += 1
                        
            # Append to CSV (Disk)
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([gen + 1, best_fitness, avg_fitness, generations_without_improvement])
            
            # --- CATACLYSM / PERTURBATION STRATEGY ---
            if generations_without_improvement >= 50:
                # print(f"   -> Stagnation detected (Gen {gen+1}). Triggering Perturbation!")
                
                # Keep Elites (Top 2)
                new_pop = [copy.deepcopy(x[1]) for x in scored_pop[:elitism_size]]
                
                # Strategy: 50% Mutated Clones of Best, 50% Random
                num_mutated = (pop_size - elitism_size) // 2
                best_ind = scored_pop[0][1]
                
                for _ in range(num_mutated):
                    clone = copy.deepcopy(best_ind)
                    self._scramble_mutation(clone) # Shake it up
                    self._mutate(clone)
                    new_pop.append(clone)
                
                while len(new_pop) < pop_size:
                    new_pop.append(random.sample(cities, len(cities)))
                
                population = new_pop
                generations_without_improvement = 0 
                continue 

            # Standard Evolution
            if (gen + 1) % 50 == 0:
                print(f"Gen {gen+1}/{self.generations} - Best: {best_fitness:,.2f} | Avg: {avg_fitness:,.2f}")

            new_pop = [copy.deepcopy(x[1]) for x in scored_pop[:elitism_size]]
            
            while len(new_pop) < pop_size:
                candidates = random.sample(scored_pop, tournament_size)
                p1 = min(candidates, key=lambda x: x[0])[1]
                candidates = random.sample(scored_pop, tournament_size)
                p2 = min(candidates, key=lambda x: x[0])[1]
                
                child = self._ox1(p1, p2)
                if random.random() < current_mutation_rate:
                    self._mutate(child)
                new_pop.append(child)
            
            population = new_pop

        # RECONSTRUCTION
        best_logical_path = self._reconstruct_logical_path(best_genome)        
        physical_path = self._reconstruct_physical_path(best_logical_path)
        
        return physical_path

    def _reconstruct_logical_path(self, genome):
        logical_path = []
        current_node = 0
        current_weight = 0.0
        trip_cost = 0.0
        
        for next_city in genome:
            gold_at_next = self.gold_map_list[next_city]
            d_next = self.dist_matrix[current_node][next_city]
            b_next = self.beta_matrix[current_node][next_city]
            
            cost_to_next = d_next + ((self.alpha * current_weight) ** self.beta) * b_next if current_weight > 0 else d_next
            new_weight = current_weight + gold_at_next
            
            d_ret_aft = self.dist_matrix[next_city][0]
            b_ret_aft = self.beta_matrix[next_city][0]
            cost_ret_after = d_ret_aft + ((self.alpha * new_weight) ** self.beta) * b_ret_aft if new_weight > 0 else d_ret_aft
            
            d_ret_now = self.dist_matrix[current_node][0]
            b_ret_now = self.beta_matrix[current_node][0]
            cost_ret_now = d_ret_now + ((self.alpha * current_weight) ** self.beta) * b_ret_now if current_weight > 0 else d_ret_now
            
            cost_out_empty = self.dist_matrix[0][next_city]
            cost_ret_new = d_ret_aft + ((self.alpha * gold_at_next) ** self.beta) * b_ret_aft if gold_at_next > 0 else d_ret_aft
            
            scenario_extend = trip_cost + cost_to_next + cost_ret_after
            scenario_split = trip_cost + cost_ret_now + cost_out_empty + cost_ret_new
            
            if scenario_extend <= scenario_split:
                trip_cost += cost_to_next
                current_weight = new_weight
                current_node = next_city
                logical_path.append(next_city)
            else:
                logical_path.extend([0, next_city])
                current_node = next_city
                current_weight = gold_at_next
                trip_cost = cost_out_empty
                
        logical_path.append(0)
        return logical_path

    def _reconstruct_physical_path(self, logical_path):
        physical_path = []
        current_node = 0
        
        for target in logical_path:
            if target == current_node: continue
            segment = self.path_cache[current_node][target]
            for node in segment[1:]:
                gold_val = 0.0
                if node == target and node != 0:
                    gold_val = float(round(self.gold_map_list[node], 2))
                physical_path.append((int(node), gold_val))
            current_node = target
        
        if not physical_path or physical_path[-1][0] != 0:
            physical_path.append((0, 0.0))
        return physical_path

    def _ox1(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b] = p1[a:b]
        child_set = set(p1[a:b])
        p2_ptr = 0
        for i in range(size):
            if child[i] == -1:
                while p2[p2_ptr] in child_set:
                    p2_ptr += 1
                val = p2[p2_ptr]
                child[i] = val
                child_set.add(val)
        return child

    def _mutate(self, ind):
        """Standard Inversion Mutation"""
        size = len(ind)
        a, b = sorted(random.sample(range(size), 2))
        ind[a:b+1] = ind[a:b+1][::-1]

    def _scramble_mutation(self, ind):
        """Scramble Mutation: Randomly shuffles a segment"""
        size = len(ind)
        a, b = sorted(random.sample(range(size), 2))
        sub = ind[a:b+1]
        random.shuffle(sub)
        ind[a:b+1] = sub