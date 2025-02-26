module Latul

using NamedGraphs
using Graphs
using Distances
using ITensors

export paths_by_length, correlations_GNN_edges, correlations_GNN_edges_X, reorder_GNN, on_site_szs_from_samples, NN_NNN_szs_from_samples, on_site_szs_per_samples, NN_NNN_szs_per_samples

function paths_by_length(nx::Int,ny::Int)
    # getting the edges that define higher-order hoppings 
    a1, a2 = Array{Float64,1}([1.0,0.0]), Array{Float64,1}([0.0,1.0])
    coordinates = Array{Float64,2}(undef,(nx*ny,2))
    ite = 1
    for y in 0:ny-1
        for x in 0:nx-1
            coordinates[ite,:] = x.*a1 + y.*a2
            ite += 1
        end
    end

    edges_arr = []
    for idx_i in 0:size(coordinates)[1]-1
        ii = coordinates[idx_i+1,:]
        for idx_j in 0:size(coordinates)[1]-1
            jj = coordinates[idx_j+1,:]
            dist = norm(jj-ii)
            push!(edges_arr,tuple(idx_i::Int,idx_j::Int,dist::Float64))
        end
    end

    paths_by_length = Dict{Float64,Array{Tuple{Int,Int,Float64},1}}()
    for ee in edges_arr
        path_length = round(ee[3],digits=8)
        if !haskey(paths_by_length,path_length)
            paths_by_length[path_length] = []
        end
        push!(paths_by_length[path_length],tuple(ee[1]::Int,ee[2]::Int,path_length::Float64))
    end

    sorted_paths = sort(collect(keys(paths_by_length)))

    return sorted_paths, paths_by_length
end

function nn_correlations(g::NamedGraph, ψ::MPS)
    out = Dict(zip(edges(g), [0.0 for e in edges(g)]))
    C = correlation_matrix(ψ, "Z", "Z")
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        v1_linear_index, v2_linear_index = findfirst(v -> v == v1, vertices(g)), findfirst(v -> v == v2, vertices(g))
        out[e] = C[v1_linear_index, v2_linear_index]
    end

    return out
end
  
function correlations_GNN_edges(g::NamedGraph, ψ::MPS, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})
    out = Dict(zip(NN_NNN_edges, [0.0 for e in NN_NNN_edges]))
    C = correlation_matrix(ψ, "Z", "Z")
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        v1_linear_index, v2_linear_index = findfirst(v -> v == v1, vertices(g)), findfirst(v -> v == v2, vertices(g))
        out[e] = C[v1_linear_index, v2_linear_index]
    end

    return out
end

function correlations_GNN_edges_X(g::NamedGraph, ψ::MPS, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})
    out = Dict(zip(NN_NNN_edges, [0.0 for e in NN_NNN_edges]))
    C = correlation_matrix(ψ, "X", "X")
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        v1_linear_index, v2_linear_index = findfirst(v -> v == v1, vertices(g)), findfirst(v -> v == v2, vertices(g))
        out[e] = C[v1_linear_index, v2_linear_index]
    end

    return out
end
  
function reorder_GNN(vals::Vector{Float64}, edgs::Vector{Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}}, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})
    vals_GNN = zeros(0)
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        indx = findall(x->x==tuple(v1,v2),edgs)[1]
        append!(vals_GNN, vals[indx])
        # println("edge = $(e), tuple = ($(v1),$(v2)), indx = $(indx), distances = $(vals[indx])")
    end
    return vals_GNN
end

function on_site_szs_from_samples(g::NamedGraph, samples::Vector)
    vs = collect(vertices(g))
    nsamples = length(samples)
    sigmazs = Dict(zip(vs, zeros(length(vs))))
    for v in vs
        i = findfirst(==(v), vs)
        for sample in samples
            sigmazs[v] += sample[i] == 1 ? 1 : -1
        end
        sigmazs[v] /= nsamples
    end
    return sigmazs
end

function NN_NNN_szs_from_samples(g::NamedGraph, samples::Vector, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})
    vs = collect(vertices(g))
    nsamples = length(samples)
    sigmazs = Dict(zip(NN_NNN_edges, zeros(length(NN_NNN_edges))))
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        i, j = findfirst(v -> v == v1, vs), findfirst(v -> v == v2, vs)
        for sample in samples
            sigmazs[e] += sample[i]*sample[j] == 2 ? -1 : 1
        end
        sigmazs[e] /= nsamples
    end

    return sigmazs
end

function on_site_szs_per_samples(g::NamedGraph, smpl::Vector{Int64})
    vs = collect(vertices(g))
    sigmazs = Dict(zip(vs, zeros(length(vs))))
    for v in vs
        i = findfirst(==(v), vs)
        sigmazs[v] += smpl[i] == 1 ? 1 : -1
    end
    return sigmazs
end

function NN_NNN_szs_per_samples(g::NamedGraph, smpl::Vector{Int64}, NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})

    vs = collect(vertices(g))
    sigmazs = Dict(zip(NN_NNN_edges, zeros(length(NN_NNN_edges))))
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        i, j = findfirst(v -> v == v1, vs), findfirst(v -> v == v2, vs)
        sigmazs[e] = smpl[i]*smpl[j] == 2 ? -1 : 1
    end

    return sigmazs
end

# function nearest_neighbor_szs_from_samples(g::NamedGraph, samples::Vector)
#     es = edges(g)
#     vs = collect(vertices(g))
#     nsamples = length(samples)
#     sigmazs = Dict(zip(es, zeros(length(es))))
#     for e in es
#         i, j = findfirst(==(src(e)), vs), findfirst(==(dst(e)), vs)
#         for sample in samples
#             sigmazs[e] += sample[i]*sample[j] == 2 ? -1 : 1
#         end
#         sigmazs[e] /= nsamples
#     end

#     return sigmazs
# end
  

end