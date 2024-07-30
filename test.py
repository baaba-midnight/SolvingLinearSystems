import numpy as np
import systemsSolver
from systemsSolver import graph

inconsistent = np.array([
	[1, 2, 3, 1],
	[4, 5, 6, 2],
	[7, 8, 9, 4],
])

inconsistentR2 = np.array([
	[2, 3, 5],
	[4, 6, 11],
])

consistent_unique = np.array([
	[2, 1, -1, 8],
	[-3, -1, 2, -11],
	[-2, 1, 2, -3]
])

consistent_uniqueR2 = np.array([
	[2, 3, 5],
	[4, 1, 1],
])

one_free = np.array([
	[2, 3, 4, 5],
	[5, 6, 7, 8],
	[8, 9, 10, 11],
])

one_freeR2 = np.array([
	[1, 2, 4],
	[2, 4, 8],
])

two_free = np.array([
	[10, -3, 2, 9],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
])

jemima = np.array([
	[1,2,3,4,5,0],
	[2,4,6,8,10,0],
	[3,2,3,4,5,0],
])

systems = [inconsistent, consistent_unique, one_free, two_free, inconsistentR2, consistent_uniqueR2, one_freeR2, jemima]
system_names = ["Example 1: Inconsistent System",
				"Example 2: Conistent System w unique solution",
				"Example 3: Consistent System w one free variable",
				"Example 4: Consistent System w two free variables",
				"Example 5: Inconsistent System R2",
				"Example 6: Conistent System w unique solution R2",
				"Example 7: Consistent System w free variable R2",
				"Example 8: Jemima's 3X6"]

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
