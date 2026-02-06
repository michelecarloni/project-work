import math
import random
import time
import numpy as np
from scipy.spatial import KDTree

class ThiefSolver:
    def __init__(self, problem):
        self.problem = problem
        self.num_cities = len(problem.graph.nodes)
        self.alpha_weight = problem.alpha
        self.beta_weight = problem.beta
        
        # --- FAST DATA EXTRACTION ---
        G = problem.graph
        self.coords = np.array([G.nodes[i]['pos'] for i in range(self.num_cities)])
        self.golds = np.array([G.nodes[i]['gold'] for i in range(self.num_cities)])
        
        # --- PRECOMPUTE NEIGHBORS (KD-Tree) ---
        # Used for Nearest Neighbor initialization
        self.kdtree = KDTree(self.coords)
        self.dist_matrix = {} # Cache for distances to avoid re-calc

    def get_dist(self, u, v):
        """Fast distance with caching"""
        if u > v: u, v = v, u
        if (u, v) in self.dist_matrix:
            return self.dist_matrix[(u, v)]
        
        d = self.coords[u] - self.coords[v]
        dist = math.sqrt(d[0]**2 + d[1]**2)
        self.dist_matrix[(u, v)] = dist
        return dist

    def getSolution(self):
        start_time = time.time()
        TIME_LIMIT = 90 
        
        # 1. INITIALIZATION
        # Generate a few greedy solutions and pick the best one to start ILS
        current_tour = self.get_nearest_neighbor_tour()
        
        # Evaluate initial
        current_plan, current_cost = self.split(current_tour)
        
        best_tour = list(current_tour)
        best_cost = current_cost
        best_plan = current_plan
        
        # ILS Loop
        iteration = 0
        max_no_improve = 50
        no_improve_count = 0
        
        while (time.time() - start_time) < TIME_LIMIT:
            iteration += 1
            
            # --- STEP 1: PERTURBATION (Kick) ---
            if iteration > 1:
                candidate_tour = self.double_bridge_move(current_tour)
            else:
                candidate_tour = list(current_tour)
            
            # --- FIX: Calculate cost immediately here ---
            # This ensures candidate_cost is always defined before the comparison
            _, candidate_cost = self.split(candidate_tour)
            
            # --- STEP 2: LOCAL SEARCH (2-Opt) ---
            improved = True
            ls_iter = 0
            MAX_LS_STEPS = 300 
            
            while improved and ls_iter < MAX_LS_STEPS:
                improved = False
                ls_iter += 1
                
                # Try random 2-opt moves
                for _ in range(20):
                    i = random.randint(1, self.num_cities - 4)
                    j = random.randint(i + 2, self.num_cities - 2)
                    
                    new_tour = candidate_tour[:i+1] + candidate_tour[i+1:j+1][::-1] + candidate_tour[j+1:]
                    
                    # Check Cost
                    _, new_cost = self.split(new_tour)

                    if new_cost < candidate_cost:
                        candidate_tour = new_tour
                        candidate_cost = new_cost
                        improved = True
                        break # Accept first improvement
                
            # --- STEP 3: ACCEPTANCE ---
            if candidate_cost < current_cost:
                current_tour = candidate_tour
                current_cost = candidate_cost
                current_plan, _ = self.split(current_tour)
                no_improve_count = 0
                
                # Check Global Best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = list(current_tour)
                    best_plan = current_plan
                    # print(f"Iter {iteration}: New Best Cost = {best_cost:,.2f}")
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    current_tour = self.get_nearest_neighbor_tour(random_start=True)
                    _, current_cost = self.split(current_tour)
                    no_improve_count = 0
                    
        return best_plan, best_cost

    def double_bridge_move(self, tour):
        """
        A 'Kick' operator that disrupts the tour more than a 2-opt/3-opt 
        but preserves the 4 segments' internal structure.
        Good for escaping local optima in TSP.
        """
        # Tour indices: 0 ... N-1
        # We need 4 cuts: a < b < c < d
        n = len(tour)
        if n < 8: return tour # Too small
        
        pos = sorted(random.sample(range(1, n - 1), 3))
        a, b, c = pos
        
        # Segments:
        # S1: 0...a
        # S2: a...b
        # S3: b...c
        # S4: c...end
        # Reconnect as: S1 - S4 - S3 - S2 (Example perturbation)
        
        s1 = tour[:a]
        s2 = tour[a:b]
        s3 = tour[b:c]
        s4 = tour[c:]
        
        return s1 + s4 + s3 + s2

    def get_nearest_neighbor_tour(self, random_start=False):
        """
        Constructs a greedy Nearest Neighbor tour.
        If random_start=True, picks a random city as "start" (after base).
        """
        visited = {0}
        tour = [0]
        curr = 0
        
        # If random start, pick the first node randomly
        if random_start:
            first = random.randint(1, self.num_cities - 1)
            tour.append(first)
            visited.add(first)
            curr = first
            
        for _ in range(self.num_cities - len(visited)):
            # Fast lookup with KDTree
            dists, indices = self.kdtree.query(self.coords[curr], k=20)
            
            found = False
            for idx in indices:
                if idx not in visited:
                    next_node = idx
                    found = True
                    break
            
            if not found:
                remaining = list(set(range(self.num_cities)) - visited)
                next_node = remaining[0]
            
            tour.append(next_node)
            visited.add(next_node)
            curr = next_node
            
        return tour

    def split(self, tour):
        """
        Split Algorithm: Cuts the giant tour into optimal trips.
        """
        cities = tour[1:] # Cities to visit
        n = len(cities)
        
        V = np.full(n + 1, float('inf'))
        V[0] = 0
        parent = np.zeros(n + 1, dtype=int)
        
        alpha = self.alpha_weight
        beta = self.beta_weight
        golds = self.golds
        
        # Heuristic Window Size
        # If Beta is high, trips are short (2-5 cities). 
        # If Beta is low, trips can be long (20+ cities).
        if beta > 1.5:
            WINDOW = 6
        elif beta > 1.0:
            WINDOW = 12
        else:
            WINDOW = 25
            
        for j in range(n):
            if V[j] == float('inf'): continue
            
            current_gold = 0
            current_cost = 0
            u = 0 # Start at Base
            
            # Forward trip construction
            for k in range(1, WINDOW + 1):
                if j + k > n: break
                
                v = cities[j + k - 1]
                
                # Travel u -> v
                d = self.get_dist(u, v)
                
                if current_gold > 0:
                    current_cost += d + (alpha * d * current_gold) ** beta
                else:
                    current_cost += d
                
                current_gold += golds[v]
                
                # Return v -> Base
                d_return = self.get_dist(v, 0)
                return_cost = d_return + (alpha * d_return * current_gold) ** beta
                
                total = current_cost + return_cost
                
                target = j + k
                if V[j] + total < V[target]:
                    V[target] = V[j] + total
                    parent[target] = j
                
                u = v
        
        # Reconstruct
        chunks = []
        curr = n
        while curr > 0:
            prev = parent[curr]
            segment = cities[prev : curr]
            chunk = []
            for node in segment:
                chunk.append((node, golds[node]))
            chunk.append((0, 0))
            chunks.append(chunk)
            curr = prev
            
        chunks.reverse()
        flat_path = [step for sublist in chunks for step in sublist]
        
        return flat_path, V[n]