import numpy as np
import systemsSolver
from systemsSolver import graph

presentation = np.array([
	[4,8,12,0],
	[1,2,3,6],
	[2,4,6,0],
])

systems = [presentation]
system_names = ["Example 1: Class Presentation"]

for system in systems:
	print("System")
	print(system)
	print("--------------------------------------------------")
	results = systemsSolver.gauss_jordan_partial_pivot(system)
	print("RREF")
	print(results)
	print("---------------------------------------------------")
	print("---------------------------------------------------")
	solution_vector, basis_vectors, translation_vector = systemsSolver.get_solution_set(results)
	graph(solution_vector, basis_vectors, translation_vector)
