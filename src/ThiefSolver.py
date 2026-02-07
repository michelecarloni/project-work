import networkx as nx
import numpy as np
import random
import copy

class ThiefSolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.num_nodes = len(self.graph.nodes)
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Pre-calculate data for efficiency
        self.gold_map = {node: self.graph.nodes[node]['gold'] for node in self.graph.nodes}
        self.dist_matrix, self.beta_matrix = self._calculate_complex_matrices()
        
        # GA Hyperparameters - Scaled for Problem Size
        self.pop_size = 200 if self.num_nodes <= 100 else 100
        self.generations = 1000 if self.num_nodes <= 100 else 500
        self.mutation_rate = 0.2
        self.elitism_size = 2

    def _calculate_complex_matrices(self):
        """
        Pre-calculates edge-wise distances for the non-linear cost formula.
        Uses Dijkstra to ensure we sum d^beta correctly along the physical shortest path.
        """
        d_mat = np.zeros((self.num_nodes, self.num_nodes))
        b_mat = np.zeros((self.num_nodes, self.num_nodes))
        path_gen = nx.all_pairs_dijkstra_path(self.graph, weight='dist')

        for source, paths_dict in path_gen:
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

    def _step_cost(self, d_linear, d_beta, weight):
        """
        Calculates cost for a path segment.
        Formula: Cost = d_linear + (alpha * weight)^beta * d_beta
        """
        if weight == 0: return d_linear
        return d_linear + ((self.alpha * weight) ** self.beta) * d_beta

    def _evaluate(self, genome):
        """
        Greedy Split Heuristic:
        Iterates through the 'logical' genome (permutation of cities).
        Decides whether to extend the current trip or return to base (0) 
        based on which option is locally cheaper.
        """
        total_cost = 0.0
        current_node = 0
        current_weight = 0.0
        logical_path = [] # Stores sequence like [5, 12, 0, 3, ...]
        trip_cost = 0.0

        for next_city in genome:
            gold_at_next = self.gold_map[next_city]
            
            # Cost to GO to next city
            cost_to_next = self._step_cost(self.dist_matrix[current_node][next_city], 
                                           self.beta_matrix[current_node][next_city], current_weight)
            new_weight = current_weight + gold_at_next
            
            # Cost if we were to return AFTER next city
            cost_return_after = self._step_cost(self.dist_matrix[next_city][0], 
                                                self.beta_matrix[next_city][0], new_weight)
            
            # Alternative: Return NOW, then go to next city
            cost_return_now = self._step_cost(self.dist_matrix[current_node][0], 
                                              self.beta_matrix[current_node][0], current_weight)
            cost_out_empty = self.dist_matrix[0][next_city] # Weight is 0
            cost_return_new = self._step_cost(self.dist_matrix[next_city][0], 
                                              self.beta_matrix[next_city][0], gold_at_next)

            # Compare: (Extend Trip) vs (Split Trip)
            scenario_extend = trip_cost + cost_to_next + cost_return_after
            scenario_split = trip_cost + cost_return_now + cost_out_empty + cost_return_new

            if scenario_extend <= scenario_split:
                # Extend
                trip_cost += cost_to_next
                current_weight = new_weight
                current_node = next_city
                logical_path.append(next_city)
            else:
                # Split (Return to base)
                total_cost += (trip_cost + cost_return_now)
                # We record 0 (base) and then the next city
                logical_path.extend([0, next_city])
                current_node = next_city
                current_weight = gold_at_next
                trip_cost = cost_out_empty

        # Close the final loop (return to base)
        total_cost += (trip_cost + self._step_cost(self.dist_matrix[current_node][0], 
                                                   self.beta_matrix[current_node][0], current_weight))
        logical_path.append(0)
        return total_cost, logical_path

    def _reconstruct_path(self, logical_path):
        """
        Converts logical stops into the final physical path.
        Strategy: Target-Only Pickup.
        We only pick up gold when we reach the specific 'target' city dictated by the GA.
        Intermediate cities are visited (recorded in path) but NOT looted yet.
        """
        physical_path = []
        current_node = 0
        
        for target in logical_path:
            if target == current_node:
                continue
            
            # Get physical segment (e.g., [current, transit, target])
            segment = nx.shortest_path(self.graph, source=current_node, target=target, weight='dist')
            
            # Iterate segment starting from index 1 to skip duplicate start node
            for node in segment[1:]:
                gold_val = 0.0
                
                # CRITICAL FIX: Only pick up gold if this node is the *current target*.
                # We treat all other nodes as "transit only" (0.0 gold) for now.
                # They will be picked up later when they become the target.
                if node == target and node != 0:
                    gold_val = float(round(self.gold_map[node], 2))
                
                physical_path.append((int(node), gold_val))
            
            current_node = target

        # Ensure path ends at (0, 0.0)
        if not physical_path or physical_path[-1][0] != 0:
            physical_path.append((0, 0.0))
            
        return physical_path

    def _calculate_true_cost(self, physical_path):
        """
        Recalculates the exact cost of the physical path to ensure accuracy.
        """
        total_cost = 0.0
        current_weight = 0.0
        current_node = 0 # Start at base
        
        for next_node, gold_collected in physical_path:
            dist = self.problem.graph[current_node][next_node]['dist']
            
            # Calculate cost for this single edge
            d_beta = dist ** self.beta
            
            cost = self._step_cost(dist, d_beta, current_weight)
            total_cost += cost
            
            # Update state
            current_weight += gold_collected
            if next_node == 0:
                current_weight = 0.0 # Reset weight at base
            
            current_node = next_node
            
        return total_cost

    def getSolution(self):
        """
        Main execution method.
        Returns: (path_list, total_cost)
        """
        cities = list(range(1, self.num_nodes))
        population = [random.sample(cities, len(cities)) for _ in range(self.pop_size)]
        
        best_fitness = float('inf')
        best_logical_path = []

        # Evolution Loop
        for gen in range(self.generations):
            scored_pop = []
            
            # Evaluation
            for ind in population:
                cost, l_path = self._evaluate(ind)
                scored_pop.append((cost, ind, l_path))
            
            # Sort by cost (lowest is best)
            scored_pop.sort(key=lambda x: x[0])
            
            # Elitism Update
            if scored_pop[0][0] < best_fitness:
                best_fitness = scored_pop[0][0]
                best_logical_path = scored_pop[0][2]

            # Selection & Crossover
            new_pop = [copy.deepcopy(x[1]) for x in scored_pop[:self.elitism_size]]
            
            while len(new_pop) < self.pop_size:
                # Tournament
                p1 = min(random.sample(scored_pop, 5), key=lambda x: x[0])[1]
                p2 = min(random.sample(scored_pop, 5), key=lambda x: x[0])[1]
                
                child = self._ox1(p1, p2)
                
                if random.random() < self.mutation_rate:
                    self._mutate(child)
                
                new_pop.append(child)
            
            population = new_pop

        # Final Reconstruction
        physical_path = self._reconstruct_path(best_logical_path)
        true_cost = self._calculate_true_cost(physical_path)
        
        return physical_path, true_cost

    def _ox1(self, p1, p2):
        """Order Crossover 1 with O(1) optimization."""
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b] = p1[a:b]
        
        # Optimization: Use set for O(1) lookups to handle N=1000
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
        """Inversion Mutation"""
        a, b = sorted(random.sample(range(len(ind)), 2))
        ind[a:b+1] = ind[a:b+1][::-1]