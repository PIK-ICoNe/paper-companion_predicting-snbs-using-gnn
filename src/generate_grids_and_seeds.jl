using Pkg
Pkg.activate(string(@__DIR__, "/../../"))


using EmbeddedGraphs
using SyntheticNetworks
using StatsBase
using LinearAlgebra
using NLsolve
using HDF5
using LightGraphs


const N = 100 # number of nodes
const runs_grids = 10000
const K = 9.0 # coupling coefficient


struct static_flow
    static_result::Array{Float64}
    static_result_converged::Bool
end

function graph_generation(n)
    n0, p, q, r, s = 1, 1 / 5, 3 / 10, 1 / 3, 1 / 10 # parameters from Deciphering the imprint of topology on nonlinear dynamical network stability
        ## u is not chosen reasonably, because the used version SyntheticNetworks does not support using u
    u = 0
    g = generate_graph(RandomPowerGrid(n, n0, p, q, r, s, u))
    return g
end

    # residual of static flow needed to compute the static flow
function resid_static_flow!(res, input, L, P)
    V = exp.(input .* im)
    flow = real.(V .* conj(L * V))
    @. res = P + flow
end

    # computation of the static flow
function get_static_flow(g, n, L, P, R)
    x0 = imag.(R * P)
    result = nlsolve((res, x) -> resid_static_flow!(res, x, L, P), x0)
    return static_flow(mod2pi.(result.zero .- result.zero[1] .+ pi) .- pi, result.f_converged)
end

    # generating grid and static solution
function generate_static_testcase(N)
    fix_point = false
    while fix_point == false
        g = graph_generation(N)
        P = ones(N)
        idx = sample(1:N, N ÷ 2, replace=false)
        P[idx] .= -1
        L = K * laplacian_matrix(g) .* im
        static_flow_data = get_static_flow(g, N, L, P, pinv(Matrix(L)))
        if static_flow_data.static_result_converged
            fix_point = true
            return g, P, L, static_flow_data
        end
    end
end


    # function to parallelize process
function simulate_grid(r)
    seed_index = r
    id = lpad(r, length(digits(runs_grids)), '0')
    g, P, L, static_flow_data = generate_static_testcase(N)
    file_name = string("./grids/grid_", id, ".h5")
    h5open(file_name, "w") do file
        write(file, "grids_weights_col_ptr", EmbeddedGraphs.weights(g).colptr)
        write(file, "grids_weights_row_val", EmbeddedGraphs.weights(g).rowval)
        # Store vertex position as 2-D Array
        write(file, "grids_vertices", permutedims(hcat(g.vertexpos...)))
        write(file, "P", P)
        write(file, "static_result", static_flow_data.static_result)
        write(file, "seed_index", seed_index)
    end
end

for r in 1:runs_grids
    simulate_grid(r)
end

## information about stored variables
# grids_vertices: x and y coordiantes of nodes, second index: node, third index: x- or y-coordiante
# grids_weights_col_ptr and grids_weights_row_val: weights of edges stored in matrices using compressed storage
# P: sources/sinks,  index represents node
# static_result_stored: index represents node, so there is a vector with phase angle at each node
