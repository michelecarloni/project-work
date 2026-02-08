# testing solution function

from Problem import Problem
from s338856 import solution

N: int = 100
d: float = 0.2
a: float = 1
b: float = 1

problem: Problem = Problem(num_cities=N, density=d, alpha=a, beta=b)

print("Testing solution function...")

result = solution(problem)

print()
print("RESULTS")
print(result)