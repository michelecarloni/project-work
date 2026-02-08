import networkx as nx
import numpy as np
import random
import copy
import os
import csv


# EVALUATOR OF THE FITNESS: it takes the 'split' or 'extend' strategy
def fast_evaluate(genome, dist_matrix, beta_matrix, gold_map, alpha, beta):
    """
    Calculates the fitness (total cost) of a given genome (city sequence).
    
    Strategy: greedy lookahead (1-step)
    At each city, the thief decides whether to:
    - EXTEND: carry the gold to the next city. (used when the physical distance is short or the gold is small)
    - SPLIT:  return to base to drop gold, then go to next city. (used when the next city is far or the gold is large)
    
    Args:
    - genome: List of city indices representing the order of visitation.
    - dist_matrix: pre-computed matrix of physical distances.
    - beta_matrix: pre-computed matrix of distance^beta (for weighted cost).
    - gold_map: list of gold amounts at each city.
    - alpha, beta: problem parameters.
        
    Returns:
    - total_cost: the calculated cost of the path.
    - path: Empty list (unused here, kept for signature compatibility).
    """

    total_cost = 0.0
    current_node = 0        # starting point is the main city (index 0)
    current_weight = 0.0    # current gold being carried
    trip_cost = 0.0         # cost accumulated in the current trip  
    
    for next_city in genome:
        gold_at_next = gold_map[next_city]
        
        # --- PRE CALCULATE COSTS FOR DECISION MAKING

        d_next = dist_matrix[current_node][next_city]   # Extract the distance from current_node to next_city
        b_next = beta_matrix[current_node][next_city]   # Extract the pre-computed distance^beta from current_node to next_city
        
        # inline cost calculationfor the EXTEND scenario
        if current_weight == 0:
            cost_to_next = d_next
        else:
            cost_to_next = d_next + ((alpha * current_weight) ** beta) * b_next

        # hypothetical weight if we pick up the gold at next_city
        new_weight = current_weight + gold_at_next
        
        # COMPUTE cost to return to base after picking up gold at next_city
        d_ret_aft = dist_matrix[next_city][0]
        b_ret_aft = beta_matrix[next_city][0]
        
        if new_weight == 0:
            cost_ret_after = d_ret_aft
        else:
            cost_ret_after = d_ret_aft + ((alpha * new_weight) ** beta) * b_ret_aft
            
        # COMPUTE cost to return to base now (before picking up gold at next_city)
        d_ret_now = dist_matrix[current_node][0]
        b_ret_now = beta_matrix[current_node][0]

        if current_weight == 0:
            cost_ret_now = d_ret_now
        else:
            cost_ret_now = d_ret_now + ((alpha * current_weight) ** beta) * b_ret_now
        
        # Costo to go directly from base to next_city (if we were to split and drop gold first)
        cost_out_empty = dist_matrix[0][next_city]
        
        # Cost to return to Base after visiting next_city (if we had dropped gold before)
        # In this scenario, we only carry gold_at_next
        if gold_at_next == 0:
            cost_ret_new = d_ret_aft
        else:
            cost_ret_new = d_ret_aft + ((alpha * gold_at_next) ** beta) * b_ret_aft

        # ---GREEDY DECISION: opt A or opt B

        # opt A: extend the current trip (carry gold to next city)
        scenario_extend = trip_cost + cost_to_next + cost_ret_after
        # opt B: split the trip (return to base, then go to next city)
        scenario_split = trip_cost + cost_ret_now + cost_out_empty + cost_ret_new

        if scenario_extend <= scenario_split:
            # It's cheaper to carry the gold and keep going
            trip_cost += cost_to_next
            current_weight = new_weight
            current_node = next_city
        else:
            # It's cheaper to return to base first, drop the gold, then go to next city
            total_cost += (trip_cost + cost_ret_now)
            current_node = next_city
            current_weight = gold_at_next
            trip_cost = cost_out_empty

    # Final return to main city
    d_final = dist_matrix[current_node][0]
    b_final = beta_matrix[current_node][0]
    
    if current_weight == 0:
        total_cost += (trip_cost + d_final)
    else:
        total_cost += (trip_cost + d_final + ((alpha * current_weight) ** beta) * b_final)
        
    return total_cost, []

class ThiefSolver:
    """
    This class encapsulate the whole genetic algorithm for solving the thief Problem.
    Uses a hybrid approach:
    - GA optimizes the order of cities.
    - Greedy Heuristic for deciding whether using the EXTEND or SPLIT strategy.
    """

    def __init__(self, problem, output_dir="logs"):
        # Initialization of all the important attributes for the class
        self.problem = problem
        self.graph = problem.graph
        self.num_nodes = len(self.graph.nodes)
        self.density = nx.density(self.graph)
        self.alpha = self.problem.alpha
        self.beta = self.problem.beta

        self.output_dir = output_dir
        
        # OPTIMIZATION 1: cache gold values
        # Accessing list[i] is much faster than graph.nodes[i]['gold']
        self.gold_map_list = [0.0] * self.num_nodes
        for i in range(self.num_nodes):
            self.gold_map_list[i] = self.graph.nodes[i]['gold']

        # OPTIMIZATION 2: Path Cache
        # Stores the full sequence of nodes for every shortest path (u, v)
        self.path_cache = {} 

        print("Initializing Matrices & Caching Paths (Standard Dijkstra)...")
        # Pre-compute distance matrices to allow O(1) cost lookups during evolution
        d_mat_np, b_mat_np = self._calculate_complex_matrices_and_cache()
        
        self.dist_matrix = d_mat_np.tolist()
        self.beta_matrix = b_mat_np.tolist()
        
        # GA PARAMETERS
        # the populations size and the number of generations are adapted
        # based on the problem size. Done for reducing computational time
        self.pop_size = 200 if self.num_nodes <= 100 else 150 
        self.generations = 1000 if self.num_nodes <= 100 else 800
        self.mutation_rate = 0.2
        self.elitism_size = 2
        self.tournament_size = 5

    def _calculate_complex_matrices_and_cache(self):
        """
        Runs All-Pairs Dijkstra to:
        - cache the physical path (sequence of nodes) for every pair of cities.
        - pre-calculate the 'linear distance' (L) and 'beta distance' (L^Beta).
        
        This allows the fitness function to calculate weighted costs instantly 
        without traversing the graph, even for multi-hop paths.
        (Again done for efficiency)
        """

        d_mat = np.zeros((self.num_nodes, self.num_nodes))
        b_mat = np.zeros((self.num_nodes, self.num_nodes))
        
        # This generator yields (source, {target: [path_nodes]})
        path_gen = nx.all_pairs_dijkstra_path(self.graph, weight='dist')

        for source, paths_dict in path_gen:
            # save the cache: the index is the city and the value is a dict {target_city: [path_nodes]}
            self.path_cache[source] = paths_dict
            
            # From the source city we look at every single possible target city
            # we might want to visit and we calculate the total distance and the total distance^beta of the path
            for target, path in paths_dict.items():
                if source == target: continue
                l_sum, b_sum = 0.0, 0.0
                # calculate path metrics segment by segment
                for i in range(len(path) - 1):
                    d = self.graph[path[i]][path[i+1]]['dist']
                    l_sum += d
                    b_sum += d ** self.beta
                d_mat[source][target] = l_sum
                b_mat[source][target] = b_sum
                
        return d_mat, b_mat

    def getSolution(self):
        """
        Main execution method for the Genetic Algorithm.
        
        Returns:
        - physical_path: List of tuples [(node_id, gold_taken), ...]
        """

        # Initialize Population (Random Permutations of Cities)
        cities = list(range(1, self.num_nodes))
        population = [random.sample(cities, len(cities)) for _ in range(self.pop_size)]
        
        best_fitness = float('inf')
        best_genome = None
        
        generations_without_improvement = 0
        current_mutation_rate = self.mutation_rate

        # Unpack local variables for loop speed optimization
        pop_size = self.pop_size
        elitism_size = self.elitism_size
        tournament_size = self.tournament_size
        dist_matrix = self.dist_matrix
        beta_matrix = self.beta_matrix
        gold_map = self.gold_map_list
        alpha = self.alpha
        beta = self.beta

        # CSV CONFIG
        csv_filename = f"prob_{self.num_nodes}_{round(self.density, 1)}_{round(self.alpha, 1)}_{round(self.beta, 1)}.csv"
        csv_dir = self.output_dir
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_path = os.path.join(csv_dir, csv_filename)
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Best_Cost", "Avg_Cost", "Stagnation_Count"])


        # --- MAIN EVOLUTION LOOP
        for gen in range(self.generations):
            scored_pop = []
            sum_fitness = 0.0
            
            # 1 - evaluation step: calculate fitness for each individual in the population
            for ind in population:
                cost, _ = fast_evaluate(ind, dist_matrix, beta_matrix, gold_map, alpha, beta)
                scored_pop.append((cost, ind))
                sum_fitness += cost
            
            # sort by cost in ascending order
            scored_pop.sort(key=lambda x: x[0])
            avg_fitness = sum_fitness / pop_size
            
            # 2 - Check for improvement and update best solution
            if scored_pop[0][0] < best_fitness:
                best_fitness = scored_pop[0][0]
                best_genome = copy.deepcopy(scored_pop[0][1])
                generations_without_improvement = 0
                current_mutation_rate = self.mutation_rate
            else:
                generations_without_improvement += 1
                        
            # 3 - append generation stats to CSV
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([gen + 1, best_fitness, avg_fitness, generations_without_improvement])
            
            # 4 - cataclysm strategy: Check for stagnation and apply perturbation if needed
            # If stuck for 50 generations, shake up the population to escape local optima
            if generations_without_improvement >= 50:
                # print(f"   -> Stagnation detected (Gen {gen+1}). Triggering Perturbation!")
                
                # Keep Elites (Top 2)
                new_pop = [copy.deepcopy(x[1]) for x in scored_pop[:elitism_size]]
                
                # Strategy: 50% heavily mutated clones of Best, 50% pure Random
                num_mutated = (pop_size - elitism_size) // 2
                best_ind = scored_pop[0][1]
                
                for _ in range(num_mutated):
                    clone = copy.deepcopy(best_ind)
                    self._scramble_mutation(clone)      # major structural change
                    self._mutate(clone)                 # fine-tuning mutation: minor change
                    new_pop.append(clone)
                
                while len(new_pop) < pop_size:
                    new_pop.append(random.sample(cities, len(cities)))
                
                population = new_pop
                generations_without_improvement = 0 
                continue 

            # print statement for monitoring progress
            if (gen + 1) % 50 == 0:
                print(f"Gen {gen+1}/{self.generations} - Best: {best_fitness:,.2f} | Avg: {avg_fitness:,.2f}")

            # 5 - selection and crossover step: create the next generation
            new_pop = [copy.deepcopy(x[1]) for x in scored_pop[:elitism_size]]
            
            while len(new_pop) < pop_size:
                # tournament selection
                candidates = random.sample(scored_pop, tournament_size)
                p1 = min(candidates, key=lambda x: x[0])[1]
                candidates = random.sample(scored_pop, tournament_size)
                p2 = min(candidates, key=lambda x: x[0])[1]
                
                # perform crossover (OX1) and mutation
                child = self._ox1(p1, p2)
                if random.random() < current_mutation_rate:
                    self._mutate(child)
                new_pop.append(child)
            
            population = new_pop

        # 6 - final reconstruction of the best solution found by the GA
        best_logical_path = self._reconstruct_logical_path(best_genome)        
        physical_path = self._reconstruct_physical_path(best_logical_path)
        
        return physical_path

    def _reconstruct_logical_path(self, genome):
        """
        Re-runs the Greedy Evaluator one last time to record the actual path decisions.
        This converts the chromosome [A, B, C] into a logical path [A, 0, B, C, 0].
        
        It mirrors the logic of fast_evaluate exactly to ensure the final path 
        matches the fitness score found during evolution.
        """
        logical_path = []
        current_node = 0
        current_weight = 0.0
        trip_cost = 0.0
        
        for next_city in genome:
            # Re-calculate all costs exactly as in fast_evaluate
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
                # Extend trip
                trip_cost += cost_to_next
                current_weight = new_weight
                current_node = next_city
                logical_path.append(next_city)
            else:
                # Split trip (Drop gold at 0)
                logical_path.extend([0, next_city])
                current_node = next_city
                current_weight = gold_at_next
                trip_cost = cost_out_empty
                
        logical_path.append(0) # Always return to base at the end
        return logical_path

    def _reconstruct_physical_path(self, logical_path):
        """
        Expands the logical path (City A -> City B) into the full physical path
        (City A -> Node X -> Node Y -> City B) using the cached Dijkstra paths.
        Also assigns gold pickup events.
        """
        physical_path = []
        current_node = 0
        
        for target in logical_path:
            if target == current_node: continue

            # Retrieve detailed path from cache
            segment = self.path_cache[current_node][target]
            for node in segment[1:]:
                gold_val = 0.0
                # Only pick up gold if we are at the target city (and it's not the main city)
                if node == target and node != 0:
                    gold_val = float(round(self.gold_map_list[node], 2))
                physical_path.append((int(node), gold_val))
            current_node = target
        
        # Ensure we end at the main city with 0 gold
        if not physical_path or physical_path[-1][0] != 0:
            physical_path.append((0, 0.0))
        return physical_path

    def _ox1(self, p1, p2):
        """
        Order Crossover (OX1) operator.
        Preserves relative order of cities from parents
        without breaking the good chains of cities we have already found.

        process:
        - it copies a solid slice of cities from parent 1 to the Child.
        (This preserves a 'good chain' of connections found by parent 1).
        - it fills the remaining spots using the relative order from parent 2.
        (This ensures valid permutations without creating duplicates).
        """

        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size

        # Inherit the exact sub-tour (chain of cities) from parent 1
        child[a:b] = p1[a:b]
        child_set = set(p1[a:b])

        p2_ptr = 0

        # fill the gaps in the child with cities from parent 2
        for i in range(size):
            if child[i] == -1:
                while p2[p2_ptr] in child_set:
                    p2_ptr += 1
                val = p2[p2_ptr]
                child[i] = val
                child_set.add(val)
        return child

    def _mutate(self, ind):
        """Standard Inversion Mutation: reverses a sub-segment of the genome."""
        size = len(ind)
        a, b = sorted(random.sample(range(size), 2))
        ind[a:b+1] = ind[a:b+1][::-1]

    def _scramble_mutation(self, ind):
        """Scramble Mutation: Randomly shuffles a sub-segment (Used for Perturbation)."""
        size = len(ind)
        a, b = sorted(random.sample(range(size), 2))
        sub = ind[a:b+1]
        random.shuffle(sub)
        ind[a:b+1] = sub