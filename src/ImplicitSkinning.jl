# __precompile__()

module ImplicitSkinning

using SymPy

function grad(f, x)
    [diff(f, x[i]) for i in 1:length(x)]
end

function hrbf_field(x, phi, vs, lambdas, betas)
    num_points = length(lambdas)
    @assert size(vs, 1) == length(lambdas)
    @assert size(betas, 1) == length(lambdas)
    @assert num_points >= 1

    d = 0 * lambdas[1]
    for i = 1:num_points
        phi_expr = phi(norm(x - vec(vs[i,:])))
        d += lambdas[i] * phi_expr + dot(vec(betas[i,:]), grad(phi_expr, x))[1]
    end
    d
end

type HRBFGenerator
	dimension::Integer
	num_points::Integer
	A::Array{Function, 2}
	d_lambda::Function
end

function HRBFGenerator(dimension::Integer, num_points::Integer, phi::Function)
	lambdas = Sym[symbols("lambda$(i)", real=true) for i in 1:num_points]
	x = Sym[symbols("x$(i)", real=true) for i in 1:dimension]
	vs = Sym[symbols("v$(j)_$(i)", real=true) for j in 1:num_points, i in 1:dimension]
	betas = Sym[symbols("beta$(j)_$(i)", real=true) for j in 1:num_points, i in 1:dimension]

	d = hrbf_field(x, phi, vs, lambdas, betas)
	A_generator = Array{Function}(num_points * (dimension + 1), num_points * (dimension + 1))

	row = 1
	for i = 1:num_points
	    col = 1
	    d_at_v_i = d([x[k] => vs[i,k] for k = 1:dimension]...)
	    for j = 1:num_points
	        @assert expand(diff(d_at_v_i, lambdas[j])) == expand(phi(norm(vs[i,:] - vs[j,:])))
	        A_generator[row, col] = lambdify(diff(d_at_v_i, lambdas[j]), vs[:])
	        col += 1
	    end
	    for j = 1:length(betas)
	        A_generator[row, col] = lambdify(diff(d_at_v_i, betas'[j]), vs[:])
	        col += 1
	    end
	#     for j = 1:num_points
	#         for k = 1:dimension
	#             A_generator[row, col] = lambdify(diff(d_at_v_i, betas[j,k]), vs[:])
	#             col += 1
	#         end
	#     end
	    row += 1
	end

	g = grad(d, x)

	for i = 1:num_points
	    for k = 1:dimension
	        col = 1
	        for j = 1:num_points
	            expr = diff(g[k], lambdas[j])([x[l] => vs[i,l] for l = 1:dimension]...)
	            if expr == 0.0 || isnan(expr)
	                A_generator[row, col] = (x...) -> 0.0
	            else
	                A_generator[row, col] = lambdify(expr, vs[:])
	            end
	            col += 1
	        end
	        for j = 1:length(betas)
	            expr = diff(g[k], betas'[j])([x[l] => vs[i,l] for l = 1:dimension]...)
	            if expr == 0.0 || isnan(expr)
	                A_generator[row, col] = (x...) -> 0.0
	            else
	                A_generator[row, col] = lambdify(expr, vs[:])
	            end
	            col += 1
	        end
	#         for j = 1:num_points
	#             for l = 1:dimension
	#                 expr = diff(g[k], betas[j,l])([x[l] => vs[i,l] for l = 1:dimension]...)
	#                 if expr == 0.0 || isnan(expr)
	#                     A_generator[row, col] = (x...) -> 0.0
	#                 else
	#                     A_generator[row, col] = lambdify(expr, vs[:])
	#                 end
	#                 col += 1
	#             end
	#         end
	        row += 1
	    end
	end

	d_lambda = lambdify(d, vcat(x, vs[:], lambdas, betas[:]))
	HRBFGenerator(dimension, num_points, A_generator, d_lambda)
end

function get_field(gen::HRBFGenerator, points, normals)
	dimension = size(points, 2)
	num_points = size(points, 1)
    A = similar(gen.A, Float64)
    for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            A[i,j] = gen.A[i,j](points...)
        end
    end

    b = vcat(zeros(num_points), normals'[:])
    y = A \ b
    lambda_values = y[1:num_points]
    beta_values = reshape(y[num_points+1:end], dimension, num_points)'
    return (x...) -> gen.d_lambda(vcat(collect(x), points[:], lambda_values, beta_values[:])...)
end

end # module
