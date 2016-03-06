using ImplicitSkinning
using Base.Test

function hrbf_2d()
	points = [1. 0; 0 1; -1 0; 0 -1]'
	normals = [1. 1; 0 1; -1 1; 0 -1]'
	dimension = 2
	num_points = size(points, 1)

	# phi_degree = 3

	# gen = ImplicitSkinning.HRBFGenerator(dimension, x -> x^3)

	# f_result = ImplicitSkinning.get_field(gen, points, normals)

	field = HermiteRadialField(points, normals, ImplicitSkinning.rbf_cube)

	X = linspace(-2, 2)
	Y = linspace(-2, 2)
	Z = zeros(length(X), length(Y))

	@time for i = 1:length(X)
	    for j = 1:length(Y)
	        Z[j,i] = evaluate(field, [X[i], Y[j]])
	    end
	end

	for i in 1:size(points, 2)
		@test isapprox(evaluate(field, points[:,i]), 0, atol=1e-6)
		eps = 1e-4
		nudged_point = points[:,i] + eps * normals[:,i]
		@test isapprox((grad(field, points[:,i])' * normals[:,i] * eps)[1], evaluate(field, nudged_point), atol=1e-6)
	end
end

hrbf_2d()
