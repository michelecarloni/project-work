import networkx as nx
import numpy as np
import heapq
import random
import time

class ThiefSolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # PARAMETERS
        # Adjusting the Tree shape based on graph density
        # High beta (0.8) = Star-like (fast return to base)
        # Low beta (0.4) = Spanning Tree (efficient connection of neighbors)
        density = nx.density(self.graph)
        self.pd_beta = 0.8 if density < 0.5 else 0.4
        
        # Pre-compute edge weights for speed
        self.dist_matrix = {} 

    def get_dist(self, u, v):
        if (u, v) in self.dist_matrix:
            return self.dist_matrix[(u, v)]
        d = self.graph[u][v]['dist']
        self.dist_matrix[(u, v)] = d
        return d

    def build_pd_tree(self):
        """
        Constructs a Prim-Dijkstra Tree.
        Simplifies the messy N^2 graph into a clean N-1 edge skeleton.
        """
        root = 0
        T = nx.Graph()
        visited = set([root])
        
        # Priority Queue: (priority_cost, u, v, dist_from_root)
        pq = []
        
        # Initialize PQ with edges from root
        for neighbor in self.graph[root]:
            w = self.get_dist(root, neighbor)
            # Priority = Edge + Beta * Dist_From_Root
            priority = w # dist_from_root is 0
            heapq.heappush(pq, (priority, root, neighbor, w))
            
        while len(visited) < self.graph.number_of_nodes():
            if not pq: break
            
            prio, u, v, dist_v = heapq.heappop(pq)
            
            if v in visited:
                continue
            
            # Add edge to Tree
            visited.add(v)
            T.add_edge(u, v, dist=self.get_dist(u, v))
            
            # Push new neighbors
            for w_node in self.graph[v]:
                if w_node not in visited:
                    edge_len = self.get_dist(v, w_node)
                    new_dist_from_root = dist_v + edge_len
                    
                    # The Hybrid Formula: Balance MST cost vs Dijkstra cost
                    new_priority = edge_len + (self.pd_beta * dist_v)
                    
                    heapq.heappush(pq, (new_priority, v, w_node, new_dist_from_root))
        return T

    def evaluate_hubs_approx(self, hubs, tree_paths):
        """
        Fast approximate cost function for the Hill Climber.
        Assumes we clear gold on the way back from the Hub.
        """
        total_cost = 0.0
        cleared_nodes = set()
        
        # Sort by path length: Short trips first -> clear inner nodes
        sorted_hubs = sorted(hubs, key=lambda h: len(tree_paths[h]))
        
        for hub in sorted_hubs:
            path = tree_paths[hub] # [0, a, b, ... hub]
            
            # Outbound cost (Weight = 0)
            trip_dist = 0.0
            for i in range(len(path)-1):
                trip_dist += self.get_dist(path[i], path[i+1])
            total_cost += trip_dist
            
            # Inbound cost (Weight accumulates)
            current_weight = 0.0
            
            # Walk backwards from Hub to 0
            for i in range(len(path)-1, 0, -1):
                u = path[i]      # Current node
                prev = path[i-1] # Next node (closer to 0)
                
                # Pick up gold if not yet cleared
                if u not in cleared_nodes:
                    current_weight += self.graph.nodes[u]['gold']
                    cleared_nodes.add(u)
                
                d = self.get_dist(u, prev)
                total_cost += d + (self.alpha * d * current_weight) ** self.beta
                
        return total_cost

    def getSolution(self):
        start_time = time.time()
        TIME_LIMIT = 180
        
        # 1. Build the Skeleton (PD-Tree)
        T = self.build_pd_tree()
        
        # 2. Get Tree Paths (Instant lookup)
        tree_paths = nx.single_source_shortest_path(T, 0)
        
        # 3. Initial Solution: All Leaves are Hubs
        leaves = [x for x in T.nodes() if T.degree(x) == 1 and x != 0]
        current_hubs = set(leaves)
        
        best_hubs = set(current_hubs)
        best_cost = self.evaluate_hubs_approx(best_hubs, tree_paths)
        
        nodes_list = list(T.nodes())
        nodes_list.remove(0)

        # 4. Stochastic Hill Climbing Optimization
        # We try adding internal nodes as Hubs (splitting trips)
        # or removing them (merging trips)
        while (time.time() - start_time) < TIME_LIMIT:
            
            # Create mutation candidate
            candidate_hubs = best_hubs.copy()
            
            # Randomly flip a node's status (Hub <-> Not Hub)
            # Constraint: Leaves must always be Hubs to ensure full coverage
            node_to_flip = random.choice(nodes_list)
            
            if node_to_flip in leaves:
                continue 
                
            if node_to_flip in candidate_hubs:
                candidate_hubs.remove(node_to_flip)
            else:
                candidate_hubs.add(node_to_flip)
            
            # Evaluate
            new_cost = self.evaluate_hubs_approx(candidate_hubs, tree_paths)
            
            if new_cost < best_cost:
                best_hubs = candidate_hubs
                best_cost = new_cost
        

        # 5. Format and Calculate Exact Cost
        final_path = self.format_output(best_hubs, tree_paths)
        final_cost = self.calculate_final_cost(final_path)
        

        return final_path, final_cost

    def format_output(self, hubs, tree_paths):
        # Sort hubs: Shortest paths first to clear inner gold efficiently
        sorted_hubs = sorted(list(hubs), key=lambda h: len(tree_paths[h]))
        
        final_path = []
        cleared_nodes = set()
        
        for hub in sorted_hubs:
            path = tree_paths[hub] # [0, node1, node2... hub]
            
            # --- OUTBOUND LEG (0 -> Hub) ---
            # Standard move: Go to next node, carry 0 extra gold
            for i in range(1, len(path)):
                next_node = path[i]
                final_path.append((next_node, 0.0))
            
            # --- INBOUND LEG (Hub -> 0) ---
            # Retrace steps backwards
            for i in range(len(path)-1, 0, -1):
                curr = path[i]
                next_node = path[i-1] # Parent
                
                # GOLD LOGIC:
                # We are currently AT 'curr'. We are about to move to 'next_node'.
                # Should we pick up the gold at 'curr' before we leave?
                # YES, if it hasn't been picked up by a previous trip.
                
                if curr not in cleared_nodes:
                    # We modify the LAST step in final_path.
                    # The last step was "Move to curr". 
                    # We update it to say "Move to curr AND pick up gold".
                    last_step_node, _ = final_path[-1]
                    assert last_step_node == curr
                    
                    gold = self.graph.nodes[curr]['gold']
                    final_path[-1] = (curr, gold)
                    cleared_nodes.add(curr)
                
                # Now move to the next node (towards base)
                final_path.append((next_node, 0.0))
                
        return final_path

    def calculate_final_cost(self, path):
        """Calculates exact cost iterating through the formatted path."""
        total_cost = 0.0
        current_weight = 0.0
        curr_node = 0
        
        for next_node, gold_picked in path:
            d = self.get_dist(curr_node, next_node)
            total_cost += d + (self.alpha * d * current_weight) ** self.beta
            
            if next_node == 0:
                current_weight = 0.0
            else:
                current_weight += gold_picked
            
            curr_node = next_node
            
        return total_cost