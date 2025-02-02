import numpy as np
import matplotlib.pyplot as plt


def plot3D(solution_vector, basis_vectors, translation_vector):
	"""
    Plot given solution vector, basis vector and translation vector that exist in the subspace R3

    Parameters:
    solution_vector : numpy.ndarray
                solution vector for unique systems
    basis_vector : numpy.ndarray
                the basis of the null space of the system
    translation_vector : numpy.ndarray
                the translation vector of a non-homogeneous system

    Returns:
    None
    """

	# plotting the plane
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	if solution_vector is not None and basis_vectors is not None and translation_vector is not None:
		if not np.all(np.isinf(solution_vector)):
			# unique solution p = np.array([0,0,0]), basis_vector = []
			u = solution_vector
			ax.set_title("Plot of Solution Vector, $u$")
			ax.scatter(u[0], u[1], u[2], c='green', s=100,
					   label=f'$u$ ({round(u[0], 2)},{round(u[1], 2)},{round(u[2], 2)})')
		else:
			# Non-homogenous system has a translation vector not at the origin hence must be translated
			p = translation_vector
			if basis_vectors.shape[0] == 1:
				u = basis_vectors[0]
				lim = max(np.max(np.abs(u)), np.max(np.abs(p)))
				# generate point for grid
				c = np.linspace(-lim, lim, 100)
				span_points = np.outer(c, u)

				# plotting Homogenous System
				# extract X, Y, Z coordinates for plotting
				X, Y, Z = span_points[:, 0], span_points[:, 1], span_points[:, 2]

				ax.plot(X, Y, Z, label='Homogenous System', color='blue')

				# plotting Non-homogenous case
				# plot the translation of the homogenous system
				X = X + p[0]
				Y = Y + p[1]
				Z = Z + p[2]

				ax.plot(X, Y, Z, label='Non-homogenous System', color='purple')

				# plot the basis vector and translation vector
				ax.quiver(0, 0, 0, u[0], u[1], u[2], color='green',
						  label=f'$u$ ({round(u[0], 2)},{round(u[1], 2)},{round(u[2], 2)})')
				ax.quiver(0, 0, 0, p[0], p[1], p[2], color='orange',
						  label=f'$p$ ({round(p[0], 2)},{round(p[1], 2)},{round(p[2], 2)})')

			elif basis_vectors.shape[0] == 2:
				u = basis_vectors
				lim = max(np.max(np.abs(u)), np.max(np.abs(p)))
				# generate point for grid
				c = np.linspace(-lim, lim, 100)
				d = np.linspace(-lim, lim, 100)
				C, D = np.meshgrid(c, d)

				# plane for homogeous case
				# create the plane for the vectors
				X = C * u[0, 0] + D * u[1, 0]
				Y = C * u[0, 1] + D * u[1, 1]
				Z = C * u[0, 2] + D * u[1, 2]
				ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color="blue", label="Homogenous System")

				# plane for non-homogenous case
				X = C * u[0, 0] + D * u[1, 0] + p[0]
				Y = C * u[0, 1] + D * u[1, 1] + p[1]
				Z = C * u[0, 2] + D * u[1, 2] + p[2]

				ax.plot_surface(X, Y, Z, alpha=1, rstride=100, cstride=100, color="purple",
								label="Non-Homogeneous System")

				# plot the basis vectors
				ax.quiver(0, 0, 0, u[0, 0], u[0, 1], u[0, 2], color='green',
						  label=f'u({u[0, 0]}, {u[0, 1]}, {u[0, 2]})')
				ax.quiver(0, 0, 0, u[1, 0], u[1, 1], u[1, 2], color='orange',
						  label=f'v({u[1, 0]}, {u[1, 1]}, {u[1, 2]})')

				# plot the translation vector
				ax.quiver(0, 0, 0, p[0], p[1], p[2], color='black',
						  label=f'$p$ ({round(p[0], 2)},{round(p[1], 2)},{round(p[2], 2)})')

		# plot origin
		ax.scatter(0, 0, 0, color='red', s=100, label='origin')

		# labels and show plot
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")

		ax.set_title("Solution Plot")
		ax.legend()

		plt.show()
	else:
		print("This system is inconsistent. No plot")
