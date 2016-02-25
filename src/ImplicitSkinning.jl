__precompile__()

module ImplicitSkinning

using SymPy

function grad(f, x)
    [diff(f, x[i]) for i in 1:length(x)]
end

function hrbf_field(x, phi, v, lambda, beta)
    phi_expr = phi(norm(x - v))
    return lambda * phi_expr + dot(beta, grad(phi_expr, x))[1]
end

type HRBFGenerator
	dimension::Integer
	partials::Dict{Tuple{Symbol, Symbol}, Function}
	field_func::Function
end

function HRBFGenerator(dimension::Integer, phi::Function)
	lambda = symbols("lambda", real=true)
	x = Sym[symbols("x$(i)", real=true) for i in 1:dimension]
	v = Sym[symbols("v$(i)", real=true) for i in 1:dimension]
	beta = Sym[symbols("beta$(i)", real=true) for i in 1:dimension]
	f = hrbf_field(x, phi, v, lambda, beta)
	g = grad(f, x)

	df_dlambda_func = lambdify(diff(f, lambda), [x...,v...])
	df_dbeta_func = [lambdify(diff(f, beta[i]), [x...,v...]) for i in 1:dimension]
	dg_dlambda_func = [lambdify(diff(g[i], lambda), [x...,v...]) for i in 1:dimension]
	dg_dbeta_func = [lambdify(diff(g[i], beta[j]), [x...,v...]) for i in 1:dimension, j in 1:dimension]
	f_func = lambdify(f, [lambda, beta..., x..., v...])

	partials = Dict{Tuple{Symbol, Symbol}, Function}()
	partials[(:f, :lambda)] = (point_index, lambda_index) -> (points -> df_dlambda_func(points[point_index,:]..., points[lambda_index,:]...))
	partials[(:f, :beta)] = (point_index, beta_index, coord_index) -> (points -> df_dbeta_func[coord_index](points[point_index,:]..., points[beta_index,:]...))
	partials[(:g, :lambda)] = (point_index, lambda_index, grad_coord_index) -> (points -> dg_dlambda_func[grad_coord_index](points[point_index,:]..., points[lambda_index,:]...))
	partials[(:g, :beta)] = (point_index, beta_index, grad_coord_index, beta_coord_index) -> (points -> dg_dbeta_func[grad_coord_index, beta_coord_index](points[point_index,:]..., points[beta_index,:]...))
	HRBFGenerator(dimension, partials, f_func)
end

function get_field{T}(gen::HRBFGenerator, points::Array{T}, normals::Array{T})
	dimension = gen.dimension
	@assert dimension == size(points, 2)
	@assert dimension == size(normals, 2)
	num_points = size(points, 1)

	A_type = promote_type(T, Float64)
	A_11 = A_type[gen.partials[(:f, :lambda)](i,j)(points) for i in 1:num_points, j in 1:num_points]
	A_12 = hcat([A_type[gen.partials[(:f, :beta)](i,j,k)(points) for i in 1:num_points, k in 1:dimension] for j in 1:num_points]...)
	A_21 = vcat([A_type[gen.partials[(:g, :lambda)](i, j, k)(points) for k in 1:dimension, j in 1:num_points] for i in 1:num_points]...)
	A_22 = vcat([hcat([A_type[gen.partials[(:g, :beta)](i, j, k, l)(points) for k in 1:dimension, l in 1:dimension] for j in 1:num_points]...) for i in 1:num_points]...)

	A = [A_11 A_12; A_21 A_22]
	A[map(isnan, A)] = 0.0
	b = vcat(zeros(num_points), normals'[:])
	y = A \ b
	lambda_values = y[1:num_points]
	beta_values = reshape(y[num_points+1:end], dimension, num_points)'
	return (x_values...) -> sum([gen.field_func(lambda_values[i], beta_values[i,:]..., x_values..., points[i,:]...) for i = 1:num_points])
end

end # module
