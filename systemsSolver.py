import plot2D
import plot3D
import numpy as np


def gauss_jordan_partial_pivot(A):
	"""
    Perform Gauss-Jordan elimination with partial pivoting to solve Ax = b.

    Parameters:
    A : list of lists
        Augmented matrix [A b] of shape (N, N+1) where N is the number of variables.
        Last column should contain the constants (b vector).

    Returns:
    B : numpy.ndarray
        The augmented matrix after Gauss-Jordan elimination with partial pivoting, where the last column
        contains the solution vector.

    Notes:
    - m represents num_rows - number of rows.
    - n represents num_cols - number of columns.
    - r represents leading_entry_count - number of leading entries.
    - pivot represents the index of the pivot element.
    """
	B = np.array(A, dtype=float)  # Initialize output matrix as a float array
	m, n = B.shape  # Get the number of rows and columns
	r = 0  # Initialize count of leading entries

	for k in range(n):  # Loop over the columns
		if r >= m:
			break
		# Find the pivot row (Our strategy is partial pivoting - using the largest entry in sub column)
		pivot = np.argmax(np.abs(B[r:m, k])) + r
		if np.abs(B[pivot, k]) < 1e-10:  # Check if pivot element is close to zero
			continue
		# Swap the current row with the pivot row
		B[[r, pivot]] = B[[pivot, r]]
		# Normalize the pivot row
		B[r] = B[r] / B[r, k]
		# Eliminate the current column in all other rows
		for i in range(m):
			if i != r:
				B[i] = B[i] - B[i, k] * B[r]
		r += 1  # Increment count of leading entries

	# Round very small values to zero in the RREF matrix
	B[np.abs(B) < 1e-10] = 0.0

	return B


def get_solution_set(A):
	"""
    Extract the solution vector or basis set from the RREF matrix.

    Parameters:
    A : numpy.ndarray
        RREF matrix of shape (N, N+1).

    Returns:
    tuple of 3 numpy.ndarrays
        Solution vector,
        Basis set of vectors,
        Translation vector,
        or None for all 3 if there is no solution.
    """

	N = A.shape[0]
	M = A.shape[1] - 1  # Number of variables
	solution_vector = np.full(M, float('inf'))
	basis_vectors = []
	translation_vector = np.zeros(M)

	# Check for existence
	for i in range(N):
		# If the row is all zeros except for the last element, there is no solution
		if np.count_nonzero(A[i, :-1]) == 0 and A[i, -1] != 0:
			return None, None, None

		# Check for uniqueness
		# If the row has exactly one non-zero element and the last element is non-zero, we have a unique solution for that variable
		if np.count_nonzero(A[i, :-1]) == 1:
			idx = np.argmax(A[i, :-1] != 0)
			value = A[i, -1] / A[i, idx]
			solution_vector[idx] = value

	# Count number of leading ones to aid checking of free variables
	# Identify leading ones in each row
	leading_ones = -np.ones(M, dtype=int)
	for i in range(N):
		row = A[i, :-1]
		non_zero_indices = np.where(row != 0)[0]
		if len(non_zero_indices) > 0:
			leading_ones[non_zero_indices[0]] = i

	# Check for free variables by comparing number of leading 1s to row count M
	# Identify free variables (columns without leading 1s)
	free_vars = [j for j in range(M) if leading_ones[j] == -1]

	# If there are free variables, we have a basis set of vectors
	# So we extract the basis and it's corresponding translation vector
	if len(free_vars) > 0:
		# reset solution vector to inf
		solution_vector = np.full(M, float('inf'))
		for var in free_vars:
			basis_vector = np.zeros(M)
			basis_vector[var] = 1
			for i in range(M):
				if leading_ones[i] != -1:
					basis_vector[i] = -A[leading_ones[i], var]
			basis_vectors.append(basis_vector)

		# Extract translation vector from the last column of the RREF matrix
		translation_vector = A[:, -1]

		# round extremely small values to zero
		basis_vectors = np.array(basis_vectors)
		basis_vectors[np.abs(basis_vectors) < 1e-10] = 0

		# return inf solution vector, basis and translation vector
		return np.array(solution_vector), np.array(basis_vectors), translation_vector

	# return unique solution and zero translation vector
	return np.array(solution_vector), np.array(basis_vectors), np.zeros(M)


def print_solution(matrix):
	"""
    Print the solution vector for a system of linear equations given its RREF matrix.

    Parameters:
    matrix : numpy.ndarray
        Reduced row echelon form matrix representing the system of linear equations.

    Returns:
    None
    """
	m, n = matrix.shape
	solution_vector, basis_vectors, translation_vector = get_solution_set(matrix)

	if solution_vector is None:
		print("The system has no solution.")
	elif len(basis_vectors) == 0:
		print("The system has a unique solution:")
		for idx, value in enumerate(solution_vector):
			# Round very small values close to zero to zero
			if np.abs(value) < 1e-10:
				value = 0.0
			print(f"x{idx + 1} = {value:.10g}")
	else:
		print("The system has infinitely many solutions with basis vectors:")
		for count, basis in enumerate(basis_vectors):
			print(
				f"Basis vector {count + 1}: {basis}")  # 10 significant digits in general format to print -0.99999 as -1


def graph(solution_vector, basis_vectors, translation_vector):
	"""
	Plots the graph for either vectors in the subspace of R2 and R3
	
	Parameters
	----------
	solution_vector : if unique contains the solution vector of system
	basis_vectors : holds the basis of the null space of the system
	translation_vector : stores the translation vector of the non-homogenous system
	
	Returns
	-------
	None
	"""
	# check if vector is in R2 or R3
	if (solution_vector is None) and (basis_vectors is None) and (translation_vector is None):
		print("The system is inconsistent")
	elif len(solution_vector) == 2:
		# plot in 2D
		plot2D.plot2D(solution_vector, basis_vectors, translation_vector)
	else:
		# plot in 3D
		plot3D.plot3D(solution_vector, basis_vectors, translation_vector)
