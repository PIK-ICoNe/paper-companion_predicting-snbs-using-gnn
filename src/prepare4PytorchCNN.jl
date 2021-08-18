using Pkg
Pkg.activate("../")

using HDF5
using LightGraphs, EmbeddedGraphs
using StatsBase
using LinearAlgebra



runs_grids = 10000
const grid_index_start = 1
const grid_index_end = 1000
const K = 9.

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

function get_mapping_of_nodes_using_current_flow_betweenness(g, R)
    current_flow_betweenness = zeros(length(g.vertexpos), 2)
    current_flow_betweenness[:,1] = get_current_flow_betweenness(g, R)
    current_flow_betweenness[:,2] = [i for i in 1:length(g.vertexpos)]
    sorted_nodes = sortslices(current_flow_betweenness, dims=1, rev=true)
    return @. Int(sorted_nodes[:,2])
end


function mapping_of_old_nodes(map_nodes)
    N = size(map_nodes, 1)
    map_from_old_to_new = Int.(zeros(N, 2))
    map_from_old_to_new[:,1] = map_nodes
    map_from_old_to_new[:,2] = [i for i in 1:N]
    sorted = sortslices(map_from_old_to_new, dims=1)
    return sorted[:,2]
end

function order_graph(g, R)
    sorted_nodes = get_mapping_of_nodes_using_current_flow_betweenness(g, R)
    old_nodes_mapped_to_new = mapping_of_old_nodes(sorted_nodes)
    graph_sorted = EmbeddedGraph(SimpleGraph(0), [])
    for i in 1:length(g.vertexpos)
        add_vertex!(graph_sorted, g.vertexpos[sorted_nodes[i]])
    end
    old_edges = collect(edges(g))
    for i in 1:ne(g)
        old_src = src(old_edges[i])
        old_dst = dst(old_edges[i])
        new_src = old_nodes_mapped_to_new[old_src]
        new_dst = old_nodes_mapped_to_new[old_dst]
        add_edge!(graph_sorted, new_src, new_dst)
    end
    return graph_sorted
end

function get_current_flow_betweenness(g, R)
    num_nodes =  length(g.vertexpos)
    A = K * adjacency_matrix(g)
    current_flow_betweenness = zeros(num_nodes)
    for i in 1:num_nodes
        for t in 1:num_nodes
            for s in 1:t
                for j in 1:num_nodes
                    if i != s && i != t
                        current_flow_betweenness[i] += abs(A[i,j] * (R[i,s] + R[j,t] - R[i,t] - R[j,s]))
                    end
                end
            end
        end
    end
    @. current_flow_betweenness / ((num_nodes - 1) * (num_nodes - 2))
    return current_flow_betweenness
end

# generate input data for CNN: grids and P
for r in grid_index_start:grid_index_end
    id = lpad(r, length(digits(runs_grids)), '0')
    file_name_read = string("grids/grids/grid_", id, ".h5")
    grids_vertices, grids_weights_col_ptr, grids_weights_row_val, P, = h5open(file_name_read, "r") do file
        read(file, "grids_vertices", "grids_weights_col_ptr", "grids_weights_row_val", "P")
    end
    g_unordered = regenerate_graph(grids_vertices, grids_weights_col_ptr, grids_weights_row_val)
    L_unordered =  K * laplacian_matrix(g_unordered)
    g = order_graph(g_unordered, pinv(Matrix(L_unordered)))
    L = K * laplacian_matrix(g)
    file_name_grid_data = string("4Pytorch/grid_cnn_data_", id, ".h5")
    h5open(file_name_grid_data, "w") do file
        write(file, "L", imag.(Matrix(L))) 
        write(file, "P", P)
    end
end
