__precompile__()

module ImplicitSkinning

import SymPy
export HermiteRadialField, TwiceDifferentiableFunction, evaluate, grad

type TwiceDifferentiableFunction
	f::Function
	df::Function
	ddf::Function
end

type HermiteRadialField{T}
	alphas::Vector{T}
	betas::Array{T, 2}
	points::Array{T, 2}
	phi::TwiceDifferentiableFunction
end

function TwiceDifferentiableFunction(f::Function)
	x = SymPy.symbols("x", real=true)
	f_sym = f(x)
	df_sym = diff(f_sym, x)
	ddf_sym = diff(df_sym, x)
	TwiceDifferentiableFunction(f, lambdify(df_sym, [x]),
		lambdify(ddf_sym, [x]))
end

phi(x) = x^3
dphi(x) = 3x^2
ddphi(x) = 6x
phi_x_cubed = TwiceDifferentiableFunction(phi, dphi, ddphi)

function HermiteRadialField{T}(points::Array{T, 2}, normals::Array{T, 2}, phi::TwiceDifferentiableFunction=phi_x_cubed)
	@assert size(points) == size(normals)
	dimension = size(points, 1)
	num_points = size(points, 2)

	A = Array{T}(num_points * (dimension + 1), num_points * (dimension + 1))
	b = Array{T}(num_points * (dimension + 1))

	for point_index = 1:num_points
		row = (point_index - 1) * (1 + dimension) + 1
		for k = 1:num_points
			col = (k - 1) * (1 + dimension) + 1
			u = vec(points[:,point_index] - points[:,k])
			n = norm(u)
			if n == 0
				A[row + (0:dimension), col + (0:dimension)] = 0
			else
				f = phi.f(n)
				df = phi.df(n)
				ddf = phi.ddf(n)
				df_over_n = df / n
				v = df_over_n .* u

				A[row, col] = f
				A[row, col + (1:dimension)] = v
				A[row + (1:dimension), col] = v
				A[row + (1:dimension), col + (1:dimension)] = (ddf - df_over_n) / (n^2) * (u * u')
				for i = 1:dimension
					A[row + i, col + i] += df_over_n
				end
			end
		end

		b[row] = 0
		b[row + (1:dimension)] = normals[:,point_index]
	end

	y = A \ b
	y = reshape(y, dimension + 1, num_points)
	alphas = y[1,:]
	betas = y[2:end,:]
	HermiteRadialField{T}(vec(alphas), betas, points, phi)
end

function evaluate{T}(field::HermiteRadialField{T}, x::Vector{T})
	value::T = zero(T)
	dimension = size(field.points, 1)
	u = Array{T}(dimension)
	for i = 1:size(field.points, 2)
		for j = 1:dimension
			u[j] = x[j] - field.points[j, i]
		end
		n = norm(u)
		if n > 0
			value += field.alphas[i] * field.phi.f(n) + field.phi.df(n) / n * (field.betas[:,i]' * u)[1]
		end
	end
	value
end

function grad{T}(field::HermiteRadialField{T}, x::Vector)
	dimension = size(field.points, 1)
	num_points = size(field.points, 2)
	g = zeros(T, dimension)
	for i = 1:num_points
		u = x - field.points[:,i]
		n = norm(u)

		if n > 1e-5
			uhat = u ./ n
			df = field.phi.df(n)
			ddf = field.phi.ddf(n)
			alpha_df = field.alphas[i] * df
			beta_uhat = (field.betas[:,i]' * uhat)[1]

			g += alpha_df .* uhat + beta_uhat * (ddf * uhat - u * df / n^2) + field.betas[:,i] * df / n
		end
	end
	g
end



# function grad(f, x)
#     [diff(f, x[i]) for i in 1:length(x)]
# end

# function hrbf_field(x, phi, v, lambda, beta)
#     phi_expr = phi(norm(x - v))
#     return lambda * phi_expr + dot(beta, grad(phi_expr, x))[1]
# end

# type HRBFGenerator
# 	dimension::Integer
# 	partials::Dict{Tuple{Symbol, Symbol}, Function}
# 	field_func::Function
# end

# function HRBFGenerator(dimension::Integer, phi::Function)
# 	lambda = symbols("lambda", real=true)
# 	x = Sym[symbols("x$(i)", real=true) for i in 1:dimension]
# 	v = Sym[symbols("v$(i)", real=true) for i in 1:dimension]
# 	beta = Sym[symbols("beta$(i)", real=true) for i in 1:dimension]
# 	f = hrbf_field(x, phi, v, lambda, beta)
# 	g = grad(f, x)

# 	df_dlambda_func = lambdify(diff(f, lambda), [x...,v...])
# 	df_dbeta_func = [lambdify(diff(f, beta[i]), [x...,v...]) for i in 1:dimension]
# 	dg_dlambda_func = [lambdify(diff(g[i], lambda), [x...,v...]) for i in 1:dimension]
# 	dg_dbeta_func = [lambdify(diff(g[i], beta[j]), [x...,v...]) for i in 1:dimension, j in 1:dimension]
# 	f_func = lambdify(f, [lambda, beta..., x..., v...])

# 	partials = Dict{Tuple{Symbol, Symbol}, Function}()
# 	partials[(:f, :lambda)] = (point_index, lambda_index) -> (points -> df_dlambda_func(points[point_index,:]..., points[lambda_index,:]...))
# 	partials[(:f, :beta)] = (point_index, beta_index, coord_index) -> (points -> df_dbeta_func[coord_index](points[point_index,:]..., points[beta_index,:]...))
# 	partials[(:g, :lambda)] = (point_index, lambda_index, grad_coord_index) -> (points -> dg_dlambda_func[grad_coord_index](points[point_index,:]..., points[lambda_index,:]...))
# 	partials[(:g, :beta)] = (point_index, beta_index, grad_coord_index, beta_coord_index) -> (points -> dg_dbeta_func[grad_coord_index, beta_coord_index](points[point_index,:]..., points[beta_index,:]...))
# 	HRBFGenerator(dimension, partials, f_func)
# end

# function get_field{T}(gen::HRBFGenerator, points::Array{T, 2}, normals::Array{T, 2})
# 	dimension = gen.dimension
# 	@assert dimension == size(points, 2)
# 	@assert dimension == size(normals, 2)
# 	num_points = size(points, 1)

# 	A_type = promote_type(T, Float64)
# 	A_11 = A_type[gen.partials[(:f, :lambda)](i,j)(points) for i in 1:num_points, j in 1:num_points]
# 	A_12 = hcat([A_type[gen.partials[(:f, :beta)](i,j,k)(points) for i in 1:num_points, k in 1:dimension] for j in 1:num_points]...)
# 	A_21 = vcat([A_type[gen.partials[(:g, :lambda)](i, j, k)(points) for k in 1:dimension, j in 1:num_points] for i in 1:num_points]...)
# 	A_22 = vcat([hcat([A_type[gen.partials[(:g, :beta)](i, j, k, l)(points) for k in 1:dimension, l in 1:dimension] for j in 1:num_points]...) for i in 1:num_points]...)

# 	A = [A_11 A_12; A_21 A_22]
# 	A[map(isnan, A)] = 0.0
# 	b = vcat(zeros(num_points), normals'[:])
# 	y = A \ b
# 	lambda_values = y[1:num_points]
# 	beta_values = reshape(y[num_points+1:end], dimension, num_points)'
# 	return (x_values...) -> sum([gen.field_func(lambda_values[i], beta_values[i,:]..., x_values..., points[i,:]...) for i = 1:num_points])
# end

# global generator_cache 

# function get_field{T}(points::Array{T, 2}, normals::Array{T, 2}; phi_degree::Int=3)
# 	dimension = size(points, 2)
# 	global generator_cache
# 	if !in(generator_cache, (dimension, phi_degree))
# 		generator_cache[(dimension, phi_degree)] = HRBFGenerator(dimension, x -> x^phi_degree)
# 	end
# 	gen = generator_cache[(dimension, phi_degree)]
# 	get_field(gen, points, normals)
# end

# function __init__()
# 	global generator_cache = Dict{Tuple{Int, Int}, HRBFGenerator}()
# end

end # module
