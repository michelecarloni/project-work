from Problem import Problem
from src.ThiefSolver import ThiefSolver

def solution(p:Problem):
    """
    Standard interface for the project-work challenge.
    Returns: [(c1, g1), (c2, g2), ..., (0, 0)]
    """
    solver = ThiefSolver(problem=p)

    path = solver.getSolution()
    return path