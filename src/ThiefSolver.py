import math
import random
import time
import numpy as np
from scipy.spatial import KDTree

class ThiefSolver:
    """
    Solves the Travelling Thief Problem using Iterated Local Search (ILS).
    The thief can make multiple trips back to the main city to deposit gold.
    
    The algorithm works in two phases:
    1. Tour Construction: Creates a giant tour visiting all cities using nearest neighbor heuristic
    2. Split Algorithm: Optimally divides the giant tour into multiple trips that return to base
    """
    
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
        self.dist_matrix = {}  # Cache for distances to avoid re-calc

    def get_dist(self, u, v):
        """Fast distance with caching"""
        if u > v:
            u, v = v, u
        if (u, v) in self.dist_matrix:
            return self.dist_matrix[(u, v)]
        
        d = self.coords[u] - self.coords[v]
        dist = math.sqrt(d[0]**2 + d[1]**2)
        self.dist_matrix[(u, v)] = dist
        return dist

    def filter_profitable_cities(self):
        """
        Filter out cities where the cost of visiting might exceed the benefit.
        For very high beta/alpha, some cities may not be worth visiting.
        Returns set of potentially profitable city indices.
        """
        if self.beta_weight < 1.5 or self.num_cities < 500:
            # For small problems or low beta, visit all cities
            return set(range(self.num_cities))
        
        profitable = {0}  # Base is always included
        for city in range(1, self.num_cities):
            # Simple heuristic: estimate if gold value justifies the trip cost
            dist = self.get_dist(0, city)
            gold = self.golds[city]
            # Rough cost estimate for round trip with gold
            approx_cost = 2 * dist + (self.alpha_weight * dist * gold) ** self.beta_weight
            # If gold seems valuable relative to cost, keep it
            if gold > approx_cost / 50:  # Lenient threshold
                profitable.add(city)
        
        # Ensure we visit at least some cities (even if marginally profitable)
        if len(profitable) < max(10, self.num_cities // 10):
            return set(range(self.num_cities))
        
        return profitable

    def getSolution(self):
        """
        Main solver method that returns the optimal path and total cost.
        
        Returns:
            tuple: (path, total_cost) where path is a list of (city, gold) tuples
                   ending with (0, 0) to indicate return to base
        """
        start_time = time.time()
        TIME_LIMIT = 10
        
        # Filter cities for difficult problems
        self.profitable_cities = self.filter_profitable_cities()
        
        # 1. INITIALIZATION - Try multiple initial solutions
        initial_solutions = []
        
        # Solution 1: Standard nearest neighbor
        tour1 = self.get_nearest_neighbor_tour()
        plan1, cost1 = self.split(tour1)
        initial_solutions.append((tour1, plan1, cost1))
        
        # Solution 2: Random start nearest neighbor (if time allows)
        if self.num_cities > 100:
            tour2 = self.get_nearest_neighbor_tour(random_start=True)
            plan2, cost2 = self.split(tour2)
            initial_solutions.append((tour2, plan2, cost2))
        
        # Select best initial solution
        current_tour, current_plan, current_cost = min(initial_solutions, key=lambda x: x[2])
        
        # Evaluate initial solution using split algorithm
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
            
            # Calculate cost after perturbation
            _, candidate_cost = self.split(candidate_tour)
            
            # --- STEP 2: LOCAL SEARCH (2-Opt) ---
            improved = True
            ls_iter = 0
            # Adaptive iterations based on problem size
            MAX_LS_STEPS = min(500, 100 + self.num_cities // 5)
            
            while improved and ls_iter < MAX_LS_STEPS:
                improved = False
                ls_iter += 1
                
                # Try random 2-opt moves
                for _ in range(20):
                    i = random.randint(1, self.num_cities - 4)
                    j = random.randint(i + 2, self.num_cities - 2)
                    
                    # Perform 2-opt: reverse segment between i and j
                    new_tour = candidate_tour[:i+1] + candidate_tour[i+1:j+1][::-1] + candidate_tour[j+1:]
                    
                    # Check Cost
                    _, new_cost = self.split(new_tour)

                    if new_cost < candidate_cost:
                        candidate_tour = new_tour
                        candidate_cost = new_cost
                        improved = True
                        break  # Accept first improvement
                
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
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Restart with a new random solution
                    current_tour = self.get_nearest_neighbor_tour(random_start=True)
                    _, current_cost = self.split(current_tour)
                    no_improve_count = 0
                    
        return best_plan, best_cost

    def double_bridge_move(self, tour):
        """
        A 'Kick' operator that disrupts the tour more than a 2-opt/3-opt 
        but preserves the 4 segments' internal structure.
        Good for escaping local optima in TSP.
        
        Splits tour into 4 segments and reconnects them in a different order.
        """
        n = len(tour)
        if n < 8:
            return tour  # Too small
        
        # Select 3 random cut points
        pos = sorted(random.sample(range(1, n - 1), 3))
        a, b, c = pos
        
        # Split into 4 segments and reconnect
        s1 = tour[:a]
        s2 = tour[a:b]
        s3 = tour[b:c]
        s4 = tour[c:]
        
        return s1 + s4 + s3 + s2

    def get_nearest_neighbor_tour(self, random_start=False):
        """
        Constructs a greedy Nearest Neighbor tour.
        If random_start=True, picks a random city as "start" (after base).
        Only visits cities in self.profitable_cities if filtering is active.
        """
        visited = {0}
        tour = [0]
        curr = 0
        
        # Determine which cities to visit
        cities_to_visit = self.profitable_cities if hasattr(self, 'profitable_cities') else set(range(self.num_cities))
        
        # If random start, pick the first node randomly from profitable cities
        if random_start:
            candidates = list(cities_to_visit - {0})
            if candidates:
                first = random.choice(candidates)
                tour.append(first)
                visited.add(first)
                curr = first
            
        while len(visited) < len(cities_to_visit):
            # Fast lookup with KDTree
            dists, indices = self.kdtree.query(self.coords[curr], k=min(20, len(cities_to_visit)))
            
            found = False
            for idx in indices:
                if idx not in visited and idx in cities_to_visit:
                    next_node = idx
                    found = True
                    break
            
            if not found:
                remaining = list(cities_to_visit - visited)
                if not remaining:
                    break
                next_node = remaining[0]
            
            tour.append(next_node)
            visited.add(next_node)
            curr = next_node
            
        return tour

    def split(self, tour):
        """
        Split Algorithm: Cuts the giant tour into optimal trips.
        
        This is the key algorithm that allows multiple trips. It uses dynamic 
        programming to find the optimal way to split a giant tour into multiple 
        trips that return to the base city after collecting gold.
        
        Args:
            tour: A list of city indices representing a giant tour
            
        Returns:
            tuple: (path, total_cost) where path is formatted as 
                   [(city1, gold1), (city2, gold2), ..., (0, 0), ...]
                   with (0, 0) marking the end of each trip
        """
        cities = tour[1:]  # Cities to visit (excluding base city 0)
        n = len(cities)
        
        # Dynamic programming arrays
        V = np.full(n + 1, float('inf'))
        V[0] = 0
        parent = np.zeros(n + 1, dtype=int)
        
        alpha = self.alpha_weight
        beta = self.beta_weight
        golds = self.golds
        
        # Adaptive Window Size based on beta and problem size
        # High beta = expensive to carry gold = shorter trips needed
        # Large N = need smaller windows relative to problem size
        if beta > 1.5:
            if n > 500:  # Large problem
                WINDOW = min(4, max(2, int(n * 0.003)))
            else:
                WINDOW = 6
        elif beta > 1.0:
            WINDOW = min(12, max(6, int(n * 0.01)))
        else:
            WINDOW = min(25, max(10, int(n * 0.03)))
            
        # Dynamic programming to find optimal split
        for j in range(n):
            if V[j] == float('inf'):
                continue
            
            current_gold = 0
            current_cost = 0
            u = 0  # Start at Base
            
            # Forward trip construction
            for k in range(1, WINDOW + 1):
                if j + k > n:
                    break
                
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
        
        # Reconstruct path
        chunks = []
        curr = n
        while curr > 0:
            prev = parent[curr]
            segment = cities[prev:curr]
            chunk = []
            for node in segment:
                chunk.append((node, golds[node]))
            chunk.append((0, 0))  # Return to base
            chunks.append(chunk)
            curr = prev
            
        chunks.reverse()
        flat_path = [step for sublist in chunks for step in sublist]
        
        return flat_path, V[n]
