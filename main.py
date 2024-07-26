import systemsSolver
import numpy as np
import sympy as sp


def input_matrix():
	"""
	Takes user input for matrix
	
	Returns
	-------
	numpy array
		matrix
	"""

	# ask user for the number of rows and columns 2 rows or 3 rows
	rows = int(input("Enter the number of rows: "))
	cols = int(input("Enter the number of columns: "))

	# initialise an empty matrix
	matrix = []

	# gether input for each row
	for i in range(rows):
		# ask user to input the elements of the current row
		row_input = input(f"Enter the elements if row {i + 1}, separated by spaces: ")

		# split input string to get a list
		row = [float(x) for x in row_input.split()]

		# check if the number of elements in the row is equal to the number of columns
		if len(row) != cols:
			print("Error: The number of elements in each row must be equal to the number of columns.")
			return None

		# append the current row to the matrix
		matrix.append(row)

	# convert to numpy array
	matrix = np.array(matrix)

	return matrix, rows


def input_b(length):
	"""
	Take user input for solution, b
	
	Parameters
	----------
	length : the size of the matrix in terms of rows
	
	Returns
	-------
	numpy array
		the user solution column
	"""
	b = []
	# ask user for the number of rows and columns 2 rows or 3 rows
	for i in range(length):
		b.append([float(input(f"Enter b{i + 1}: "))])
	return b


def main():
	"""
	Takes user input and solves the given matrix for Ax=b and Ax=0
	
	Returns
	-------
	None
	"""

	# ask user for A and b
	matrix, rows = input_matrix()
	b = np.array(input_b(rows))
	if matrix is None:
		return

	matrix = np.hstack((matrix, b))

	# solve for Ax=b case
	print("Augmented Matrix (Ax=b): ")
	print(matrix)

	# get solution of Ax=b
	results = systemsSolver.gauss_jordan_partial_pivot(matrix)
	systemsSolver.print_solution(results)
	solution_vector, basis_vectors, translation_vector = systemsSolver.get_solution_set(results)

	print("\n")
	print("-----------------------------------------------------------")
	print("RREF of [A b]")
	print(results)
	print("-----------------------------------------------------------")

	# change the last column to 0
	matrix[:, -1] = 0
	print("Augmented Matrix (Ax=0): ")
	print(matrix)

	print("-----------------------------------------------------------")

	# automatically plot 3D or 2D
	systemsSolver.graph(solution_vector, basis_vectors, translation_vector)


if __name__ == '__main__':
	main()

# Example Plots
