import matplotlib.pyplot as plt
import numpy as np


def plot2D(solution_vector, basis_vector, translation_vector):
	"""
	Plot given solution vector, basis vector and translation vector that exist in the subspace R2

	Parameters:
	solution_vector: numpy.ndarray
			solution vector for unique systems
	basis_vector: numpy.ndarray
			the basis of the null space of the system
	translation_vector: numpy.ndarray
			the translation vector of a non-homogeneous system

	Returns:
	None
	"""

	if solution_vector is not None and basis_vector is not None and translation_vector is not None:
		fig, ax = plt.subplots(figsize=(8, 6))
		if not np.all(np.isinf(solution_vector)):
			u = solution_vector

			# plot origin
			ax.scatter(0, 0, label="Origin", c="blue")

			# Unique Solution
			ax.scatter(u[0], u[1], color='red',
					   label=f"Solution Vector ({round(u[0], 2)}, {round(u[1], 2)})")  # Solution point
			ax.annotate(f'Solution ({round(u[0], 2)}, {round(u[1], 2)})', (u[0], u[1]), textcoords="offset points",
						xytext=(10, -10),
						ha='center')

			ax.set_xlabel('X-axis')
			ax.set_ylabel('Y-axis')
			ax.set_title("Unique Solution")

			ax.grid(True)
			ax.legend()
			plt.show()
		else:
			u = basis_vector[0]
			p = translation_vector

			# plot origin
			ax.scatter(0, 0, label="Origin", c="blue")

			ax.quiver(0, 0, u[0], u[1], scale_units='xy', angles='xy', scale=1,
					  label=f'Basis Vector ({round(u[0], 2)}, {round(u[1], 2)})',
					  color="g")
			ax.quiver(0, 0, p[0], p[1], scale_units='xy', angles='xy', scale=1,
					  label=f'Translation Vector ({round(p[0], 2)}, {round(p[1], 2)})',
					  color="y")

			lim = max(max(abs(u)), max(abs(p)))
			c = np.linspace(-lim, lim, 100)
			span_points = np.outer(c, u)

			# extract X, Y coordinates for plotting
			X, Y = span_points[:, 0], span_points[:, 1]

			ax.plot(X, Y, label='Homogenous System: Span(u)', color='b')

			# set numpy array for translated_span_points to be in shape as span_points
			translated_span_points = np.empty_like(span_points)

			# pass through each vector and get the index and point
			for i, points in enumerate(span_points):
				# get the span of the respective points for vector
				X = points[0] + p[0]
				Y = points[1] + p[1]

				# add the vector in translated_span_points
				translated_span_points[i] = [X, Y]

			# Extract X, Y coordinates for plotting
			X, Y = translated_span_points[:, 0], translated_span_points[:, 1]

			# Plot the translated span
			ax.plot(X, Y, color='green', label='Non-homogenous System: span(u) + p')

		# ax.set_xlim(-lim - 1, lim + 1)
		# ax.set_ylim(-lim - 1, lim + 1)

		ax.set_xlabel('X-axis')
		ax.set_ylabel('Y-axis')
		ax.set_title("A$x$=b | A$x$=0")

		ax.grid(True)
		ax.legend()
		plt.show()
	else:
		print("The system is inconsistent.")
