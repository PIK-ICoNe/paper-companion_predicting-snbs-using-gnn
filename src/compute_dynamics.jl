using Pkg
Pkg.activate(string(@__DIR__,"/../.."))
using Distributed
using ClusterManagers

n_cpu = parse(Int, ARGS[1])

# addprocs(4)
addprocs(SlurmManager(n_cpu))
@everywhere begin
    using EmbeddedGraphs, LightGraphs
    using StatsBase
    using LinearAlgebra
    using OrdinaryDiffEq
    using HDF5
    using Random

    const N = 100
    const runs_grids = 10000 
    const pert_factor = 15
    const pert_per_node = 10000 
    const α = 0.1
    const K = 9.
    const grid_index_start = 1
    const grid_index_end = 0
    const end_time = 500

    # regenerate_graph a graph of type EmbeddedGraphs
    function regenerate_graph(vertices, grids_weights_col_ptr, grids_weights_row_val)
        new_graph = EmbeddedGraph(SimpleGraph(0), [])
        for i in 1:size(vertices, 1)
            add_vertex!(new_graph, vertices[i,:])
        end
        for i in 1:length(grids_weights_col_ptr)
            EmbeddedGraphs.weights(new_graph).colptr[i] = grids_weights_col_ptr[i]
        end
        for i in 1:length(grids_weights_col_ptr) - 1
            for j in
                grids_weights_col_ptr[i]:grids_weights_col_ptr[i + 1] - 1
                add_edge!(new_graph, grids_weights_row_val[j], i)
            end
        end
        return new_graph
    end

    
    # load grid an static result from stored files
    function load_static_test_case(id)
        file_name_read = string("../grids/grids/grid_", id, ".h5")
        grids_vertices, grids_weights_col_ptr, grids_weights_row_val, P, static_result, seed_index = h5open(file_name_read, "r") do file
            read(file, "grids_vertices", "grids_weights_col_ptr", "grids_weights_row_val", "P", "static_result", "seed_index")
        end
        g = regenerate_graph(grids_vertices, grids_weights_col_ptr, grids_weights_row_val)
        L = K * laplacian_matrix(g) .* im
        return g, P, L, static_result, seed_index
    end

    function generate_α_vec(n, α)
        return α * ones(n)
    end
    
    # parameter containing variables of Kuramoto model
    struct kura_param
        P::Array{Float64}
        L::Array{Complex{Float64}}
        static_result::Array{Float64}
        kuram_α::Array{Float64}
    end

    function kuramoto_model(ddu, du, u, kura_param, t)
        V = exp.(u .* im)
        flow = real.(V .* conj(kura_param.L * V))
        @. ddu = - kura_param.kuram_α .* du + kura_param.P + flow
    end



    # computation of the single node basin stability
    function calc_bs_vec(P, L, static_result, pert_u, pert_du, α)
        # Pre-allocate vectors of initial conditions
        du0 = zeros(N)
        u0 = copy(static_result)
        tspan = (0., end_time) # time span of simulation
        α_matrix = generate_α_vec(N, α) # damping factir
        kura_param_data = kura_param(P, L, static_result, α_matrix)
        kura_problem = SecondOrderODEProblem(kuramoto_model, du0, u0, tspan, kura_param_data) # initial ODE problem
        # allocate return variables
        max_simulation_time = zeros(N, pert_per_node)
        max_frequency_dev = zeros(N, pert_per_node)
        max_final_frequency = zeros(N, pert_per_node)
        for n in 1:N
            for i in 1:pert_per_node
                # In-place modifying the inital conditions
                du0[n] += pert_du[n,i] # frequency
                u0[n] += pert_u[n,i] # phase
                kura_problem = SecondOrderODEProblem(kuramoto_model, du0, u0, tspan, kura_param_data)
                sol = solve(kura_problem, Tsit5(), save_idxs=1:N, reltol=1e-6, abstol=1e-6, callback=cb, saveat=save_at_range)
                max_simulation_time[n,i] = sol.t[end]
                max_frequency_dev[n,i] = maximum(abs.(sol))
                max_final_frequency[n,i] = maximum(abs.(sol.u[end]))
                # Reset initial conditions for the next run
                du0[n] = 0
                u0[n] = static_result[n]
            end
        end
        return max_final_frequency, max_frequency_dev, max_simulation_time
    end

    function store_dynamics(id, max_final_frequency, max_frequency_dev, max_simulation_time, α) 
        file_name = string("dynamics/dynamics_", id, ".h5")
        h5open(file_name, "w") do file
            write(file, "max_final_frequency", max_final_frequency) 
            write(file, "max_frequency_dev", max_frequency_dev)
	    write(file, "max_simulation_time", max_simulation_time)
            write(file, "α", α)
        end 
    end

    # function to parallelize process
    function simulate_grid(r)
        id = lpad(r, length(digits(runs_grids)), '0')
        g, P, L, static_result, seed_index = load_static_test_case(id)
        seed = MersenneTwister(seed_index)
        pert_du = pert_factor * ((rand(seed, N, pert_per_node) .- 0.5) * 2)
        pert_u = (rand(seed, N, pert_per_node) .- 0.5) * 2pi
        max_final_frequency, max_frequency_dev, max_simulation_time = calc_bs_vec(P, L, static_result, pert_u, pert_du, α)
        store_dynamics(id, max_final_frequency, max_frequency_dev, max_simulation_time, α)
        println("my_id: ", myid(), "  run: ", r)
        return r
    end



     # callback
     ω_idxs =  range(1,stop=N)
     condition = (u, t, integrator) -> !(any(abs(ω_i) > .1 for ω_i in u[ω_idxs]) || t < 1)
     affect!   = (integrator) -> terminate!(integrator)
     cb = DiscreteCallback(condition, affect!)
     
     # save at
     save_at_range = exp10.(range(-2, stop=2.7, length=1000))
     save_at_range = vcat(save_at_range, end_time)
end

## using pmap to parallelize
@time sol = pmap(simulate_grid, grid_index_start:grid_index_end)
