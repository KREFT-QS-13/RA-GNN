module Latul

using NamedGraphs
using Graphs
using Distances
using ITensors

export paths_by_length, correlations_GNN_edges, correlations_GNN_edges_X, reorder_GNN, on_site_szs_from_samples, NN_NNN_szs_from_samples, on_site_szs_per_samples, NN_NNN_szs_per_samples

"""
    paths_by_length(nx::Int, ny::Int)

Computes all possible paths between sites in a 2D rectangular lattice of size nx × ny.

# Arguments
- `nx::Int`: Number of sites in x direction
- `ny::Int`: Number of sites in y direction

# Returns
- `Tuple` containing:
  - Vector of sorted unique path lengths
  - Dictionary mapping path lengths to arrays of tuples (src, dst, length)
  where src and dst are site indices and length is the Euclidean distance between them

# Details
Creates a 2D lattice with unit vectors a1 = [1,0] and a2 = [0,1], then calculates
all possible paths between sites using Euclidean distance. Useful for identifying
nearest-neighbor and next-nearest-neighbor interactions.
"""
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

"""
    nn_correlations(g::NamedGraph, ψ::ITensors.MPS)

Calculates Z-Z correlations between nearest-neighbor sites in the lattice.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `ψ::ITensors.MPS`: Matrix Product State representing the quantum state

# Returns
- Dictionary mapping edge tuples to their corresponding Z-Z correlation values

# Details
Computes ⟨ψ|ZᵢZⱼ|ψ⟩ for all pairs of sites (i,j) connected by an edge in the graph,
where Z is the Pauli-Z operator.
"""
function nn_correlations(g::NamedGraph, ψ::ITensors.MPS)
    out = Dict(zip(edges(g), [0.0 for e in edges(g)]))
    C = correlation_matrix(ψ, "Z", "Z")
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        v1_linear_index, v2_linear_index = findfirst(v -> v == v1, vertices(g)), findfirst(v -> v == v2, vertices(g))
        out[e] = C[v1_linear_index, v2_linear_index]
    end

    return out
end
  
"""
    correlations_GNN_edges(g::NamedGraph, ψ::ITensors.MPS, NN_NNN_edges, Julia_Python_node_mapping)

Calculates Z-Z correlations for specific edges defined in the Graph Neural Network format.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `ψ::ITensors.MPS`: Matrix Product State representing the quantum state
- `NN_NNN_edges`: Vector of tuples (i,j,d) representing edges between sites i and j with distance d
- `Julia_Python_node_mapping`: Dictionary mapping Python-style indices to Julia-style indices

# Returns
- Dictionary mapping edge tuples to their corresponding Z-Z correlation values

# Details
Similar to nn_correlations but calculates correlations only for specific edges
provided in the GNN format. Handles conversion between Python and Julia indexing.
"""
function correlations_GNN_edges(g::NamedGraph, ψ::ITensors.MPS, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})
    out = Dict(zip(NN_NNN_edges, [0.0 for e in NN_NNN_edges]))
    C = correlation_matrix(ψ, "Z", "Z")
    # FIXED here: C[v1_linear_index, v2_linear_index] = indices error without this
    vs = collect(vertices(g))
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        # Fixed here previous error: vertices(g) not vs -> C[(i,j),(k,l)]
        v1_linear_index, v2_linear_index = findfirst(v -> v == v1, vs), findfirst(v -> v == v2, vs)
        out[e] = C[v1_linear_index, v2_linear_index]
    end
    return out
end

"""
    correlations_GNN_edges_X(g::NamedGraph, ψ::ITensors.MPS, NN_NNN_edges, Julia_Python_node_mapping)

Calculates X-X correlations for specific edges defined in the Graph Neural Network format.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `ψ::ITensors.MPS`: Matrix Product State representing the quantum state
- `NN_NNN_edges`: Vector of tuples (i,j,d) representing edges between sites i and j with distance d
- `Julia_Python_node_mapping`: Dictionary mapping Python-style indices to Julia-style indices

# Returns
- Dictionary mapping edge tuples to their corresponding X-X correlation values

# Details
Similar to correlations_GNN_edges but calculates X-X correlations (⟨ψ|XᵢXⱼ|ψ⟩)
instead of Z-Z correlations. X is the Pauli-X operator.
"""
function correlations_GNN_edges_X(g::NamedGraph, ψ::ITensors.MPS, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})
    out = Dict(zip(NN_NNN_edges, [0.0 for e in NN_NNN_edges]))
    C = correlation_matrix(ψ, "X", "X")
    # FIXED here: C[v1_linear_index, v2_linear_index] = indices error without this
    vs = collect(vertices(g))
    for e in NN_NNN_edges
        v1, v2 = Julia_Python_node_mapping[e[1]], Julia_Python_node_mapping[e[2]]
        # Fixed here previous error: vertices(g) not vs -> C[(i,j),(k,l)]
        v1_linear_index, v2_linear_index = findfirst(v -> v == v1, vs), findfirst(v -> v == v2, vs)
        out[e] = C[v1_linear_index, v2_linear_index]
    end

    return out
end
  
"""
    reorder_GNN(vals, edgs, NN_NNN_edges, Julia_Python_node_mapping)

Reorders values according to the Graph Neural Network edge ordering.

# Arguments
- `vals::Vector{Float64}`: Values to be reordered
- `edgs`: Vector of edge tuples in original ordering
- `NN_NNN_edges`: Vector of edges in GNN ordering
- `Julia_Python_node_mapping`: Dictionary mapping Python-style indices to Julia-style indices

# Returns
- Vector of reordered values matching the GNN edge ordering

# Details
Used to convert quantities (like distances or coupling strengths) from the original
lattice ordering to the ordering expected by the Graph Neural Network.
"""
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

"""
    on_site_szs_from_samples(g::NamedGraph, samples::Vector)

Calculates average on-site magnetization from Monte Carlo samples.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `samples::Vector`: Vector of spin configurations from Monte Carlo sampling

# Returns
- Dictionary mapping vertices to their average magnetization values

# Details
For each site, computes the average spin value (±1) over all samples to get
the expectation value ⟨Z⟩ for that site.
"""
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

"""
    NN_NNN_szs_from_samples(g::NamedGraph, samples, NN_NNN_edges, Julia_Python_node_mapping)

Calculates nearest-neighbor and next-nearest-neighbor correlations from Monte Carlo samples.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `samples::Vector`: Vector of spin configurations from Monte Carlo sampling
- `NN_NNN_edges`: Vector of edges to calculate correlations for
- `Julia_Python_node_mapping`: Dictionary mapping Python-style indices to Julia-style indices

# Returns
- Dictionary mapping edges to their average correlation values

# Details
Computes average product of spins (⟨ZᵢZⱼ⟩) for specified pairs of sites
using Monte Carlo samples instead of exact quantum state calculations.
"""
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

"""
    on_site_szs_per_samples(g::NamedGraph, smpl::Vector{Int64})

Calculates on-site magnetization for a single Monte Carlo sample.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `smpl::Vector{Int64}`: Single spin configuration sample

# Returns
- Dictionary mapping vertices to their magnetization values (±1)

# Details
Similar to on_site_szs_from_samples but processes a single sample instead
of averaging over multiple samples.
"""
function on_site_szs_per_samples(g::NamedGraph, smpl::Vector{Int64})
    vs = collect(vertices(g))
    sigmazs = Dict(zip(vs, zeros(length(vs))))
    for v in vs
        i = findfirst(==(v), vs)
        sigmazs[v] += smpl[i] == 1 ? 1 : -1
    end
    return sigmazs
end

"""
    NN_NNN_szs_per_samples(g::NamedGraph, smpl, NN_NNN_edges, Julia_Python_node_mapping)

Calculates correlations for a single Monte Carlo sample.

# Arguments
- `g::NamedGraph`: Graph representing the lattice structure
- `smpl::Vector{Int64}`: Single spin configuration sample
- `NN_NNN_edges`: Vector of edges to calculate correlations for
- `Julia_Python_node_mapping`: Dictionary mapping Python-style indices to Julia-style indices

# Returns
- Dictionary mapping edges to their correlation values (±1)

# Details
Similar to NN_NNN_szs_from_samples but processes a single sample instead
of averaging over multiple samples. Used for individual measurements in
Monte Carlo sampling.
"""
function NN_NNN_szs_per_samples(g::NamedGraph, smpl::Vector{Int64}, 
    NN_NNN_edges::Vector{Tuple{Int64, Int64, Float64}}, Julia_Python_node_mapping::Dict{Int64, Tuple{Int64, Int64}})

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