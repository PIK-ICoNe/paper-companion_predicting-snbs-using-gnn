using Pkg
Pkg.activate("../")

using HDF5
using LightGraphs, EmbeddedGraphs
using StatsBase

runs_grids = 10000
const grid_index_start = 1
const grid_index_end = 1000
const snbs_tolerance = .1
const K = 9.

function compute_snbs(input_frequency, tolerance)
    N = size(input_frequency, 1)
    snbs = zeros(N)
    num_pert_per_node = size(input_frequency, 2)
    conv = input_frequency .< tolerance
    for n in 1:N
        snbs[n] = size(findall(conv[n,:] .== 1), 1)
    end
    snbs ./ num_pert_per_node
end

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


function generate_edge_index_attr(a)
    n = size(a, 1)
    ne = size(findall(a .!= 0), 1)
    edge_index =  Array{Int64}(undef, 2, ne)
    global edge_ind_count = 1
    edge_attr = Array{Float64}(undef, ne)
    global edge_attr_count = 1
    for i in 1:n
        for j in i + 1:n
            if a[i,j] != 0 && i != j
                edge_index[1,edge_ind_count] = i
                edge_index[2,edge_ind_count] = j
                edge_index[1,edge_ind_count + 1] = j
                edge_index[2,edge_ind_count + 1] = i
                edge_attr[edge_attr_count] = a[i,j]
                edge_attr[edge_attr_count + 1] = a[i,j]
                edge_ind_count += 2
                edge_attr_count += 2
            end
        end
    end
    return edge_index, edge_attr
end


# generate input data for GNN: grids and P
for r in grid_index_start:grid_index_end
    id = lpad(r, length(digits(runs_grids)), '0')
    file_name_read = string("grids/grids/grid_", id, ".h5")
    grids_vertices, grids_weights_col_ptr, grids_weights_row_val, P, = h5open(file_name_read, "r") do file
        read(file, "grids_vertices", "grids_weights_col_ptr", "grids_weights_row_val", "P")
    end
    g = regenerate_graph(grids_vertices, grids_weights_col_ptr, grids_weights_row_val)
    a = K * adjacency_matrix(g)
    edge_index, edge_attr = generate_edge_index_attr(a)
    file_name_grid_data = string("4Pytorch/grid_data_", id, ".h5")
    h5open(file_name_grid_data, "w") do file
        write(file, "edge_index", edge_index) 
        write(file, "edge_attr", edge_attr)
        write(file, "P", P)
    end
end

# compute SNBS and save as label
for r in grid_index_start:grid_index_end
    id = lpad(r, length(digits(runs_grids)), '0')
    file_name_read = string("dynamics/dynamics_", id, ".h5")
    max_final_frequency, max_frequency_dev = h5open(file_name_read, "r") do file
        read(file, "max_final_frequency", "max_frequency_dev")
    end
    snbs = compute_snbs(max_final_frequency, snbs_tolerance)
    file_name_label = string("4Pytorch/snbs_", id, ".h5")
    h5open(file_name_label, "w") do file
        write(file, "snbs", snbs) 
    end
end
