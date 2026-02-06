import math
import random
import time
import networkx as nx
import numpy as np
from scipy.spatial import KDTree

class ThiefSolver:
    def __init__(self, problem):
        self.problem = problem
        self.num_cities = len(problem.graph.nodes)
        self.alpha_weight = problem.alpha
        self.beta_weight = problem.beta
        
        # --- TIME SETTINGS ---
        self.time_limit = 180 # 3 minutes minus buffer
        
        # --- PRE-COMPUTATION ---
        G = problem.graph
        
        # 1. Extract Gold & Coords
        self.golds = np.array([G.nodes[i]['gold'] for i in range(self.num_cities)])
        self.coords = np.array([G.nodes[i]['pos'] for i in range(self.num_cities)])
        
        # 2. Store Edge Weights for O(1) Lookup
        self.edge_weights = {}
        for u, v, d in G.edges(data=True):
            w = d['dist']
            self.edge_weights[(u, v)] = w
            self.edge_weights[(v, u)] = w
            
        # 3. Compute Distance to Base (Node 0)
        # Essential for "Star" shapes and fixing sparse graph gaps
        try:
            self.dist_to_base = nx.single_source_dijkstra_path_length(G, 0, weight='dist')
        except:
            self.dist_to_base = {i: float('inf') for i in range(self.num_cities)}
            self.dist_to_base[0] = 0

        # 4. KD-Tree for Nearest Neighbor Initialization
        self.kdtree = KDTree(self.coords)

    def get_safe_dist(self, u, v):
        """
        Returns distance u->v. 
        If edge exists: returns direct weight.
        If missing: returns path via Base (u->0->v).
        This PREVENTS 'inf' errors in sparse graphs.
        """
        if u == v: return 0.0
        
        # Fast direct check
        if (u, v) in self.edge_weights:
            return self.edge_weights[(u, v)]
            
        # Fallback: Path via Base (Triangular approximation)
        d_u = self.dist_to_base.get(u, float('inf'))
        d_v = self.dist_to_base.get(v, float('inf'))
        
        if d_u == float('inf') or d_v == float('inf'):
            return float('inf') # Truly unreachable
            
        return d_u + d_v

    def getSolution(self):
        start_time = time.time()
        
        # --- STEP 1: TOURNAMENT INITIALIZATION ---
        # Generate diverse candidates and pick the best one
        candidates = []
        
        # Candidate A: Nearest Neighbor (Best for Low Beta)
        candidates.append(self.generate_nn_tour())
        
        # Candidate B: Radial Sort (Best for High Beta)
        # Sort cities by distance to base
        nodes = list(range(1, self.num_cities))
        nodes.sort(key=lambda x: self.dist_to_base.get(x, float('inf')))
        candidates.append([0] + nodes)
        
        # Candidate C: Random
        rand_nodes = list(range(1, self.num_cities))
        random.shuffle(rand_nodes)
        candidates.append([0] + rand_nodes)
        
        # Evaluate all candidates
        best_tour = None
        best_cost = float('inf')
        best_plan = []
        
        for cand in candidates:
            plan, cost = self.split(cand)
            if cost < best_cost:
                best_cost = cost
                best_tour = list(cand)
                best_plan = plan
                
        current_tour = list(best_tour)
        current_cost = best_cost
        
        # --- STEP 2: ITERATED LOCAL SEARCH ---
        iteration = 0
        max_no_improve = 50
        no_improve = 0
        
        while (time.time() - start_time) < self.time_limit:
            iteration += 1
            
            # A. PERTURBATION (Kick)
            # Apply Double Bridge to escape local optima
            if iteration > 1:
                cand_tour = self.double_bridge(current_tour)
            else:
                cand_tour = list(current_tour)
            
            # B. LOCAL SEARCH (Stochastic 2-Opt)
            # We try a fixed budget of random swaps. If one improves split_cost, we take it.
            # This effectively climbs the TTP Fitness Landscape.
            
            improved_ls = True
            ls_steps = 0
            MAX_LS = 150 # Keep it responsive
            
            # Cache the cost to avoid re-evaluating the start point
            _, cand_cost = self.split(cand_tour)
            
            while improved_ls and ls_steps < MAX_LS:
                improved_ls = False
                ls_steps += 1
                
                # Try batch of random swaps
                for _ in range(25):
                    i = random.randint(1, self.num_cities - 4)
                    j = random.randint(i + 2, self.num_cities - 2)
                    
                    # 2-Opt Move
                    new_tour = cand_tour[:i+1] + cand_tour[i+1:j+1][::-1] + cand_tour[j+1:]
                    
                    # Evaluate TTP Cost
                    _, new_cost = self.split(new_tour)
                    
                    if new_cost < cand_cost:
                        cand_tour = new_tour
                        cand_cost = new_cost
                        improved_ls = True
                        break # Greedily accept
            
            # C. ACCEPTANCE CRITERION
            if cand_cost < current_cost:
                current_tour = cand_tour
                current_cost = cand_cost
                current_plan, _ = self.split(current_tour)
                no_improve = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_plan = current_plan
                    best_tour = list(current_tour)
                    # print(f"Iter {iteration}: New Best {best_cost:,.2f}")
            else:
                no_improve += 1
                
                # D. RESTART MECHANISM
                # If stuck, restart from a mutated version of the BEST known solution
                if no_improve > max_no_improve:
                    current_tour = self.double_bridge(best_tour) # Soft restart
                    # Or Hard Restart: current_tour = self.generate_nn_tour(random_start=True)
                    _, current_cost = self.split(current_tour)
                    no_improve = 0
                    
        return best_plan, best_cost

    # --- CORE ALGORITHMS ---

    def split(self, tour):
        """
        Splits a giant tour into optimal trips.
        Includes robust distance handling to prevent 'inf'.
        """
        cities = tour[1:]
        n = len(cities)
        V = np.full(n + 1, float('inf'))
        V[0] = 0
        parent = np.zeros(n + 1, dtype=int)
        
        # Dynamic Window based on Beta
        # High Beta = Short trips (small window)
        # Low Beta = Long trips (large window)
        if self.beta_weight > 1.0: WINDOW = 12
        else: WINDOW = 25
        
        alpha = self.alpha_weight
        beta = self.beta_weight
        golds = self.golds
        
        for j in range(n):
            if V[j] == float('inf'): continue
            
            curr_w = 0
            curr_c = 0
            u = 0 # Base
            
            for k in range(1, WINDOW + 1):
                if j + k > n: break
                v = cities[j + k - 1]
                
                # Get Distance (Safe)
                d = self.get_safe_dist(u, v)
                
                # If d is inf, this path is impossible, stop extending
                if d == float('inf'): break
                
                if curr_w > 0: curr_c += d + (alpha * d * curr_w) ** beta
                else: curr_c += d
                
                curr_w += golds[v]
                
                # Return to Base (Safe)
                d_ret = self.get_safe_dist(v, 0)
                if d_ret == float('inf'): break
                
                ret_c = d_ret + (alpha * d_ret * curr_w) ** beta
                total = curr_c + ret_c
                
                idx = j + k
                if V[j] + total < V[idx]:
                    V[idx] = V[j] + total
                    parent[idx] = j
                
                u = v
                
        # Reconstruct
        if V[n] == float('inf'): 
            return [], float('inf')
            
        chunks = []
        curr = n
        while curr > 0:
            prev = parent[curr]
            seg = cities[prev : curr]
            chunks.append([(n, golds[n]) for n in seg] + [(0,0)])
            curr = prev
        chunks.reverse()
        return [x for c in chunks for x in c], V[n]

    def generate_nn_tour(self, random_start=False):
        """Generates a Nearest Neighbor tour"""
        visited = {0}
        tour = [0]
        curr = 0
        
        if random_start:
            # Try to pick a valid random start
            starts = [x for x in range(1, self.num_cities) if self.get_safe_dist(0, x) != float('inf')]
            if starts:
                curr = random.choice(starts)
                tour.append(curr)
                visited.add(curr)
        
        for _ in range(self.num_cities - len(visited)):
            # Fast lookup of 50 nearest
            dists, indices = self.kdtree.query(self.coords[curr], k=50)
            
            found = False
            for idx in indices:
                if idx not in visited:
                    # Verify reachability
                    if self.get_safe_dist(curr, idx) != float('inf'):
                        tour.append(idx)
                        visited.add(idx)
                        curr = idx
                        found = True
                        break
            
            if not found:
                # Fallback scan
                rem = list(set(range(self.num_cities)) - visited)
                # Just pick first one that works
                best_rem = -1
                for r in rem:
                    if self.get_safe_dist(curr, r) != float('inf'):
                        best_rem = r
                        break
                
                if best_rem != -1:
                    tour.append(best_rem)
                    visited.add(best_rem)
                    curr = best_rem
                else:
                    break # Graph disconnected?
                    
        return tour

    def double_bridge(self, tour):
        n = len(tour)
        if n < 8: return tour
        # 4-opt kick
        pos = sorted(random.sample(range(1, n-1), 3))
        a, b, c = pos
        return tour[:a] + tour[c:] + tour[b:c] + tour[a:b]