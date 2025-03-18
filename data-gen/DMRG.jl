"""
Module for performing Density Matrix Renormalization Group (DMRG) calculations on quantum spin systems,
specifically focusing on the Transverse Field Ising (TFI) model with position-dependent interactions.
"""
module DMRG

include("File_utils.jl")
include("Lattice_utils.jl")

using Distributions
using NPZ
using NamedGraphs
using Graphs
using Distances
using ITensors

using .Filul
using .Latul

export HTFI, TFI_DMRG, main_Mg_NN_NNN_δ, main_Mg_NN_NNN_δ_smpl
export rescaled_rand

function rescaled_rand(x, y)
    return x + (y - x) * rand()
end
  
function Base.:+(a::Tuple{T, T}, b::Tuple{T, T}) where T <: Number
    return (a[1] + b[1], a[2] + b[2])
end

# C6 coefficient and perfect r = 7 um
const C6 = 5420158.53

"""
    HTFI(g::NamedGraph, hs::Dict, Jzzs::Vector{Float64}, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, sites)

Construct the Transverse Field Ising (TFI) Hamiltonian for a given graph configuration.

# Arguments
- `g`: Named graph representing the lattice structure
- `hs`: Dictionary mapping vertices to their transverse field strengths
- `Jzzs`: Vector of ZZ interaction strengths for each edge
- `edgs`: Vector of edge tuples defining the connectivity
- `sites`: ITensor site indices for the quantum system

# Returns
- An MPO (Matrix Product Operator) representing the TFI Hamiltonian
"""
function HTFI(g::NamedGraph, hs::Dict, Jzzs::Vector{Float64}, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, sites)
    ampo = OpSum()
    g_vs = collect(vertices(g))
    # Run over the edges and add the ZZ term to the Hamiltonian with the appropriate interaction strength J
    for (idx,J) in enumerate(Jzzs)
        src = edgs[idx][1]
        dst = edgs[idx][2]
        i = findfirst(v -> v == src, g_vs)
        j = findfirst(v -> v == dst, g_vs)
        if !iszero(J)  
            ampo += J, "Z", i, "Z", j # Pauli Z operator, Sz = 1/2 σz, - => ferro, + => antiferro
        end
    end

    # Run over the vertices and the X field w/ strength h
    for v in g_vs
        i = findfirst(==(v), g_vs)
        hh = hs[v]
        ampo += hh, "X", i
    end
    H = MPO(ampo, sites)
    #H = noprime(H)
    return H
end
  
"""
    TFI_DMRG(g::NamedGraph, χ::Int64, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, sites; 
             nsweeps=20, hs::Union{Nothing,Dict}=nothing, Jzzs::Vector{Float64})

Perform DMRG calculation for the TFI model to find the ground state.

# Arguments
- `g`: Named graph representing the lattice structure
- `χ`: Maximum bond dimension for the MPS
- `edgs`: Vector of edge tuples defining the connectivity
- `sites`: ITensor site indices
- `nsweeps`: Number of DMRG sweeps (default: 20)
- `hs`: Dictionary of transverse field strengths (default: all 1.0)
- `Jzzs`: Vector of ZZ interaction strengths

# Returns
- The ground state as an MPS (Matrix Product State)
"""
function TFI_DMRG(g::NamedGraph, χ::Int64, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, sites; nsweeps=20, hs::Union{Nothing,Dict} = nothing, Jzzs::Vector{Float64})
    L = length(vertices(g))
    if hs === nothing
        hs = Dict(zip(vertices(g), [1.0 for e in vertices(g)]))
    end

    # Build the Hamiltonian
    H = HTFI(g, hs, Jzzs, edgs, sites)

    # Initial state for DMRG routine (all spins pointing up)
    init_state = ["Up" for _ in 1:L]
    ψ0 = ITensors.randomMPS(sites, init_state; linkdims = 2)

    # Set truncation parameters and no sweeps
    sweeps = Sweeps(nsweeps)
    maxdim!(sweeps, Tuple(min(2^(floor(Int64, 0.5*i)), χ)  for i in 1:nsweeps)...)

    # Run DMRG
    e, ψ0 = dmrg(H, ψ0, sweeps)

    println("DMRG Finished and found an energy of " * string(e))

    return ψ0
end

"""
    main_Mg_NN_NNN_δ(nx::Int, ny::Int, num_realization::Int, num_δs::Int, χDMRG::Int, R::Float64,
                     amp_R::Float64, hx::Float64, amp_delta::Float64, path_to_folder::String;
                     is_sampled=false, num_samples=1000, start_iter=0)

Main function to generate data for nearest-neighbor (NN) and next-nearest-neighbor (NNN) interactions
with varying transverse field strengths and position disorder.

# Arguments
- `nx`, `ny`: Dimensions of the lattice
- `num_realization`: Number of disorder realizations
- `num_δs`: Number of different transverse field strengths
- `χDMRG`: Maximum bond dimension for DMRG
- `R`: Nominal lattice spacing
- `amp_R`: Amplitude of position disorder
- `hx`: Central value of transverse field
- `amp_delta`: Amplitude of transverse field variation
- `path_to_folder`: Output directory path
- `is_sampled`: Whether to sample from wavefunctions (default: false)
- `num_samples`: Number of samples if sampling (default: 1000)
- `start_iter`: Starting iteration number (default: 0)
"""
function main_Mg_NN_NNN_δ(nx::Int,ny::Int,num_realization::Int,num_δs::Int,χDMRG::Int,R::Float64,amp_R::Float64,hx::Float64,
    amp_delta::Float64,path_to_folder::String;is_sampled=false,num_samples=1000,start_iter=0)

    println("The total number of realizations is $(num_δs*num_realization)!")

    # folder_name = "dataset_mps_NNN_PT/" * string(nx) * "x" * string(ny) * "/num_deltas_$(num_δs)"
    folder_name_clstr_size = path_to_folder * "/" * string(nx) * "x" * string(ny)
    folder_name = folder_name_clstr_size * "/num_deltas_$(num_δs)"
    if num_δs==1
        folder_name = folder_name * "/delta_" * string(round(hx,digits=1))
    end
    #folder_name = "dataset_mps_NNN_OutScope/" * string(nx) * "x" * string(ny) * "/num_deltas_$(num_δs)"
    @show folder_name

    largest_int = check_exists(folder_name)
    largest_int += start_iter*num_δs

    sorted_paths, pb_length = paths_by_length(nx,ny)

    NN_edges = pb_length[sorted_paths[2]]
    NNN_edges = pb_length[sorted_paths[3]]
    @show NN_edges, length(NN_edges)
    @show NNN_edges, length(NNN_edges)

    # Define the 2D grid as a named graph with vertices (i,j) where i is the row and j is the column
    # NOTE WE ARE IMPLICITING ORDERING THE MPS VERTICES AS A SNAKE THROUGH THE LATTICE. 
    # TO CHANGE THE ORDERING OF THE SITES (WHICH MIGHT HELP/HINDER THE SIMULATION) THEN WE NEED
    # TO REORDER THE LIST vertices(g) APPROPRIATELY
    g = NamedGraphs.NamedGraphGenerators.named_grid((nx, ny))

    # Needed to switch between NamedGraphs and GNN edges
    nodes = [i for i=0:length(vertices(g))-1]
    Julia_Python_node_mapping = Dict(zip(nodes, vertices(g)))
    @show Julia_Python_node_mapping

    hxs_grid = collect(range(hx-amp_delta,stop=hx+amp_delta,length=num_δs))
    @show hxs_grid
    num_vertices = length(vertices(g))

    nominal_values = Array{Tuple{Float64,Float64},2}([(i*R, j*R) for i in 0:nx-1, j in 0:ny-1])
    edges_array = Array{Tuple{Tuple{Int,Int},Tuple{Int,Int}},2}(undef,(length(nominal_values),length(nominal_values)))
    # Create the meshgrid
    x = range(1, stop=nx)  # From 0 to 5 with 6 points
    y = range(1, stop=ny)  # From 0 to 5 with 6 points
    X = repeat(x, 1, length(y))
    Y = repeat(y', length(x), 1)
    # nn = reshape(permutedims(collect(zip(X, Y))),nx*ny)
    nn = reshape(collect(zip(X, Y)),nx*ny)
    for i in 1:length(nominal_values) # y = j//ny+1, x = j%nx
        for j in 1:length(nominal_values)
            edges_array[i,j] = (nn[i],nn[j])
        end
    end
    @show nn

    # saving the true distances
    distances_nom = pairwise(
        Euclidean(),
        reshape(nominal_values,length(nominal_values)),
        reshape(nominal_values,length(nominal_values))
    )
    indices = findall(distances_nom.==0.0) # indices pointing to values 0
    mask = [CartesianIndex(i, j) for j in 1:size(edges_array, 2), i in 1:size(edges_array, 1) if !(CartesianIndex(i,j) in indices)] # same mask used everywhere
    distances_nom = distances_nom[mask]
    edges_array = edges_array[mask]
    # @show edges_array

    # nominal distances to be saved 
    δRs_nom_GNN = reorder_GNN(distances_nom, edges_array, NN_edges, Julia_Python_node_mapping)
    δRps_nom_GNN = reorder_GNN(distances_nom, edges_array, NNN_edges, Julia_Python_node_mapping)
    npzwrite(folder_name_clstr_size * "/Rs_nom.npy", δRs_nom_GNN)
    npzwrite(folder_name_clstr_size * "/Rps_nom.npy", δRps_nom_GNN)
    # @show δRs_nom_GNN

    # Get the indices for the MPS. Assume no conservation laws
    sites = siteinds("S=1/2", length(vertices(g)); conserve_qns=false)

    println("The number of available threads is : $(Sys.CPU_THREADS)")
    println("The number of threads used: $(Threads.nthreads())")
    H_gates = ops([("Ry", n, (θ=π / 2,)) for n in 1:nx*ny], sites)
    for realization in 1:num_δs:num_δs*num_realization

        perturbations = [(rescaled_rand(-amp_R,amp_R), rescaled_rand(-amp_R,amp_R)) for _ in 1:length(nominal_values)]
        perturbations = reshape(perturbations,size(nominal_values))
        pstns = nominal_values.+perturbations
        # @show reshape(pstns,length(pstns))

        ## all-to-all distances
        distances = pairwise(
            Euclidean(),
            reshape(pstns,length(pstns)),
            reshape(pstns,length(pstns))
        )

        distances = distances[mask]
        Jij = C6./(distances.^6)

        # @show distances
        # @show perturbations
        # @show edges_array
        # Apply the Hadamard gate on each qubit to transform to the X basis
        for δ_snapshot in 0:num_δs-1
            hxs = ones(num_vertices).*hxs_grid[δ_snapshot+1] #ones(length(edges(g)))*0
            hhh = Dict(zip(vertices(g), hxs))
            @show hxs_grid[δ_snapshot+1]
            # NN
            Jzzs_GNN = reorder_GNN(Jij, edges_array, NN_edges, Julia_Python_node_mapping)
            δRs_GNN = reorder_GNN(distances, edges_array, NN_edges, Julia_Python_node_mapping)
            # NNN
            Jpzzs_GNN = reorder_GNN(Jij, edges_array, NNN_edges, Julia_Python_node_mapping)
            δRps_GNN = reorder_GNN(distances, edges_array, NNN_edges, Julia_Python_node_mapping)
        
            # Now run DMRG to get the groundstate ψ0
            # println("hhh = $(hhh) and Jzzs = $(Jzzs)")
            @time ψ0 = TFI_DMRG(g, χDMRG, edges_array, sites; hs = hhh, Jzzs = Jij)
            
            if is_sampled # carry out the measurements by sampling the wave function
                # X
                ψ0_X = apply(H_gates, ψ0)
                ψ0_X = orthogonalize(ψ0_X, 1)
                samples_X = [ITensors.sample(ψ0_X) for i in 1:num_samples]
                # Z
                ψ0 = orthogonalize(ψ0, 1)
                samples = [ITensors.sample(ψ0) for i in 1:num_samples]
            end
            # @show samples_X
            # @show samples
            if is_sampled
                magnetization_dict = on_site_szs_from_samples(g,samples)
                nn_correlations_GNN_edges_dict = NN_NNN_szs_from_samples(g,samples,NN_edges,Julia_Python_node_mapping)
                nnn_correlations_GNN_edges_dict = NN_NNN_szs_from_samples(g,samples,NNN_edges,Julia_Python_node_mapping)
                magnetization_dict_X = on_site_szs_from_samples(g,samples_X)
                nn_correlations_GNN_edges_dict_X = NN_NNN_szs_from_samples(g,samples_X,NN_edges,Julia_Python_node_mapping)
                nnn_correlations_GNN_edges_dict_X = NN_NNN_szs_from_samples(g,samples_X,NNN_edges,Julia_Python_node_mapping)
            else
                magnetization_dict = Dict(zip(vertices(g), expect(ψ0, "Z")))
                nn_correlations_GNN_edges_dict = correlations_GNN_edges(g, ψ0, NN_edges, Julia_Python_node_mapping)
                nnn_correlations_GNN_edges_dict = correlations_GNN_edges(g, ψ0, NNN_edges, Julia_Python_node_mapping)
                magnetization_dict_X = Dict(zip(vertices(g), expect(ψ0, "X")))
                nn_correlations_GNN_edges_dict_X = correlations_GNN_edges_X(g, ψ0, NN_edges, Julia_Python_node_mapping)
                nnn_correlations_GNN_edges_dict_X = correlations_GNN_edges_X(g, ψ0, NNN_edges, Julia_Python_node_mapping)
            end
            @show nn_correlations_GNN_edges_dict
        
            Mg = [magnetization_dict[Julia_Python_node_mapping[node]] for node in nodes]
            NN_corrs_GNN = [nn_correlations_GNN_edges_dict[NN_edge] for NN_edge in NN_edges]
            NNN_corrs_GNN = [nnn_correlations_GNN_edges_dict[NNN_edge] for NNN_edge in NNN_edges]
            Mg_X = [magnetization_dict_X[Julia_Python_node_mapping[node]] for node in nodes]
            NN_corrs_GNN_X = [nn_correlations_GNN_edges_dict_X[NN_edge] for NN_edge in NN_edges]
            NNN_corrs_GNN_X = [nnn_correlations_GNN_edges_dict_X[NNN_edge] for NNN_edge in NNN_edges]
        
            npzwrite(folder_name * "/Rs_$(realization+δ_snapshot+largest_int).npy", δRs_GNN)
            npzwrite(folder_name * "/Rps_$(realization+δ_snapshot+largest_int).npy", δRps_GNN)
            npzwrite(folder_name * "/Jzzs_$(realization+δ_snapshot+largest_int).npy", Jzzs_GNN)
            npzwrite(folder_name * "/Jpzzs_$(realization+δ_snapshot+largest_int).npy", Jpzzs_GNN)
            npzwrite(folder_name * "/hxs_$(realization+δ_snapshot+largest_int).npy", hxs)
            npzwrite(folder_name * "/Mg_$(realization+δ_snapshot+largest_int).npy", Mg)
            npzwrite(folder_name * "/Mg_X_$(realization+δ_snapshot+largest_int).npy", Mg_X)
            npzwrite(folder_name * "/NN_corrs_$(realization+δ_snapshot+largest_int).npy", NN_corrs_GNN)
            npzwrite(folder_name * "/NNN_corrs_$(realization+δ_snapshot+largest_int).npy", NNN_corrs_GNN)
            npzwrite(folder_name * "/NN_corrs_X_$(realization+δ_snapshot+largest_int).npy", NN_corrs_GNN_X)
            npzwrite(folder_name * "/NNN_corrs_X_$(realization+δ_snapshot+largest_int).npy", NNN_corrs_GNN_X)
        end
    end      
end


"""
    main_Mg_NN_NNN_δ_smpl(nx::Int, ny::Int, num_δs::Int, χDMRG::Int, R::Float64,
                          amp_R::Float64, hx::Float64, amp_delta::Float64, path_to_folder::String;
                          num_samples=1000, start_iter=0)

Similar to main_Mg_NN_NNN_δ but specifically for generating sampled measurements from wavefunctions.
This version uses a single disorder realization but generates multiple samples for each transverse field value.

# Arguments
- `nx`, `ny`: Dimensions of the lattice
- `num_δs`: Number of different transverse field strengths
- `χDMRG`: Maximum bond dimension for DMRG
- `R`: Nominal lattice spacing
- `amp_R`: Amplitude of position disorder
- `hx`: Central value of transverse field
- `amp_delta`: Amplitude of transverse field variation
- `path_to_folder`: Output directory path
- `num_samples`: Number of samples per configuration (default: 1000)
- `start_iter`: Starting iteration number (default: 0)
"""
function main_Mg_NN_NNN_δ_smpl(nx::Int,ny::Int,num_δs::Int,χDMRG::Int,R::Float64,amp_R::Float64,hx::Float64,
    amp_delta::Float64,path_to_folder::String;num_samples=1000,start_iter=0)

    # folder_name = "dataset_mps_NNN_PT/" * string(nx) * "x" * string(ny) * "/num_deltas_$(num_δs)"
    folder_name_clstr_size = path_to_folder * "/" * string(nx) * "x" * string(ny)
    folder_name = folder_name_clstr_size * "/num_deltas_$(num_δs)"
    if num_δs==1
        folder_name = folder_name * "/delta_" * string(round(hx,digits=1))
    end
    #folder_name = "dataset_mps_NNN_OutScope/" * string(nx) * "x" * string(ny) * "/num_deltas_$(num_δs)"
    @show folder_name

    largest_int = check_exists(folder_name)
    largest_int += start_iter*num_δs

    sorted_paths, pb_length = paths_by_length(nx,ny)

    NN_edges = pb_length[sorted_paths[2]]
    NNN_edges = pb_length[sorted_paths[3]]
    @show NN_edges, length(NN_edges)
    @show NNN_edges, length(NNN_edges)

    # Define the 2D grid as a named graph with vertices (i,j) where i is the row and j is the column
    # NOTE WE ARE IMPLICITING ORDERING THE MPS VERTICES AS A SNAKE THROUGH THE LATTICE. 
    # TO CHANGE THE ORDERING OF THE SITES (WHICH MIGHT HELP/HINDER THE SIMULATION) THEN WE NEED
    # TO REORDER THE LIST vertices(g) APPROPRIATELY
    g = NamedGraphs.NamedGraphGenerators.named_grid((nx, ny))

    # Needed to switch between NamedGraphs and GNN edges
    nodes = [i for i=0:length(vertices(g))-1]
    Julia_Python_node_mapping = Dict(zip(nodes, vertices(g)))
    @show Julia_Python_node_mapping

    hxs_grid = collect(range(hx-amp_delta,stop=hx+amp_delta,length=num_δs))
    @show hxs_grid
    num_vertices = length(vertices(g))

    nominal_values = Array{Tuple{Float64,Float64},2}([(i*R, j*R) for i in 0:nx-1, j in 0:ny-1])
    edges_array = Array{Tuple{Tuple{Int,Int},Tuple{Int,Int}},2}(undef,(length(nominal_values),length(nominal_values)))
    # Create the meshgrid
    x = range(1, stop=nx)  # From 0 to 5 with 6 points
    y = range(1, stop=ny)  # From 0 to 5 with 6 points
    X = repeat(x, 1, length(y))
    Y = repeat(y', length(x), 1)
    # nn = reshape(permutedims(collect(zip(X, Y))),nx*ny)
    nn = reshape(collect(zip(X, Y)),nx*ny)
    for i in 1:length(nominal_values) # y = j//ny+1, x = j%nx
        for j in 1:length(nominal_values)
            edges_array[i,j] = (nn[i],nn[j])
        end
    end
    @show nn

    # saving the true distances
    distances_nom = pairwise(
        Euclidean(),
        reshape(nominal_values,length(nominal_values)),
        reshape(nominal_values,length(nominal_values))
    )
    indices = findall(distances_nom.==0.0) # indices pointing to values 0
    mask = [CartesianIndex(i, j) for j in 1:size(edges_array, 2), i in 1:size(edges_array, 1) if !(CartesianIndex(i,j) in indices)] # same mask used everywhere
    distances_nom = distances_nom[mask]
    edges_array = edges_array[mask]
    # @show edges_array

    # nominal distances to be saved 
    δRs_nom_GNN = reorder_GNN(distances_nom, edges_array, NN_edges, Julia_Python_node_mapping)
    δRps_nom_GNN = reorder_GNN(distances_nom, edges_array, NNN_edges, Julia_Python_node_mapping)
    npzwrite(folder_name_clstr_size * "/Rs_nom.npy", δRs_nom_GNN)
    npzwrite(folder_name_clstr_size * "/Rps_nom.npy", δRps_nom_GNN)
    # @show δRs_nom_GNN

    # Get the indices for the MPS. Assume no conservation laws
    sites = siteinds("S=1/2", length(vertices(g)); conserve_qns=false)

    println("The number of available threads is : $(Sys.CPU_THREADS)")
    println("The number of threads used: $(Threads.nthreads())")
    # Apply the Hadamard gate on each qubit to transform to the X basis
    H_gates = ops([("Ry", n, (θ=π / 2,)) for n in 1:nx*ny], sites)

    perturbations = [(rescaled_rand(-amp_R,amp_R), rescaled_rand(-amp_R,amp_R)) for _ in 1:length(nominal_values)]
    perturbations = reshape(perturbations,size(nominal_values))
    pstns = nominal_values.+perturbations
    # @show reshape(pstns,length(pstns))

    ## all-to-all distances
    distances = pairwise(
        Euclidean(),
        reshape(pstns,length(pstns)),
        reshape(pstns,length(pstns))
    )

    distances = distances[mask]
    Jij = C6./(distances.^6)

    # NN
    Jzzs_GNN = reorder_GNN(Jij, edges_array, NN_edges, Julia_Python_node_mapping)
    δRs_GNN = reorder_GNN(distances, edges_array, NN_edges, Julia_Python_node_mapping)
    # NNN
    Jpzzs_GNN = reorder_GNN(Jij, edges_array, NNN_edges, Julia_Python_node_mapping)
    δRps_GNN = reorder_GNN(distances, edges_array, NNN_edges, Julia_Python_node_mapping)

    # on single disorder for the whole Omega-hist and set of samples
    npzwrite(folder_name_clstr_size * "/Rs_effective.npy", δRs_GNN)
    npzwrite(folder_name_clstr_size * "/Rps_effective.npy", δRps_GNN)
    npzwrite(folder_name_clstr_size * "/Jzzs_effective.npy", Jzzs_GNN)
    npzwrite(folder_name_clstr_size * "/Jpzzs_effective.npy", Jpzzs_GNN)

    # @show distances
    # @show perturbations
    # @show edges_array
    for δ_snapshot in 1:num_samples:num_samples*num_δs
        δ_indx = Int((δ_snapshot-1)/num_samples)
        hxs = ones(num_vertices).*hxs_grid[δ_indx+1] #ones(length(edges(g)))*0
        hhh = Dict(zip(vertices(g), hxs))
        @show hxs, δ_snapshot, δ_indx
    
        # Now run DMRG to get the groundstate ψ0
        @time ψ0 = TFI_DMRG(g, χDMRG, edges_array, sites; hs = hhh, Jzzs = Jij)
        
        # carry out the measurements by sampling the wave function
        # X
        ψ0_X = apply(H_gates, ψ0)
        ψ0_X = orthogonalize(ψ0_X, 1)
        # Z
        ψ0 = orthogonalize(ψ0, 1)

        # exact correlators to be compared with ones sampled from the wave function
        magnetization_dict = Dict(zip(vertices(g), expect(ψ0, "Z")))
        nn_correlations_GNN_edges_dict = correlations_GNN_edges(g, ψ0, NN_edges, Julia_Python_node_mapping)
        nnn_correlations_GNN_edges_dict = correlations_GNN_edges(g, ψ0, NNN_edges, Julia_Python_node_mapping)
        magnetization_dict_X = Dict(zip(vertices(g), expect(ψ0, "X")))
        nn_correlations_GNN_edges_dict_X = correlations_GNN_edges_X(g, ψ0, NN_edges, Julia_Python_node_mapping)
        nnn_correlations_GNN_edges_dict_X = correlations_GNN_edges_X(g, ψ0, NNN_edges, Julia_Python_node_mapping)

        Mg = [magnetization_dict[Julia_Python_node_mapping[node]] for node in nodes]
        NN_corrs_GNN = [nn_correlations_GNN_edges_dict[NN_edge] for NN_edge in NN_edges]
        NNN_corrs_GNN = [nnn_correlations_GNN_edges_dict[NNN_edge] for NNN_edge in NNN_edges]
        Mg_X = [magnetization_dict_X[Julia_Python_node_mapping[node]] for node in nodes]
        NN_corrs_GNN_X = [nn_correlations_GNN_edges_dict_X[NN_edge] for NN_edge in NN_edges]
        NNN_corrs_GNN_X = [nnn_correlations_GNN_edges_dict_X[NNN_edge] for NNN_edge in NNN_edges]

        npzwrite(folder_name * "/Mg_Exact_$(δ_snapshot+largest_int).npy", Mg)
        npzwrite(folder_name * "/Mg_X_Exact_$(δ_snapshot+largest_int).npy", Mg_X)
        npzwrite(folder_name * "/NN_corrs_Exact_$(δ_snapshot+largest_int).npy", NN_corrs_GNN)
        npzwrite(folder_name * "/NNN_corrs_Exact_$(δ_snapshot+largest_int).npy", NNN_corrs_GNN)
        npzwrite(folder_name * "/NN_corrs_X_Exact_$(δ_snapshot+largest_int).npy", NN_corrs_GNN_X)
        npzwrite(folder_name * "/NNN_corrs_X_Exact_$(δ_snapshot+largest_int).npy", NNN_corrs_GNN_X)
        
        for smpl in 0:num_samples-1

            ψ0_smpl = ITensors.sample(ψ0)
            ψ0_X_smpl = ITensors.sample(ψ0_X)
        
            magnetization_dict = on_site_szs_per_samples(g,ψ0_smpl)
            nn_correlations_GNN_edges_dict = NN_NNN_szs_per_samples(g,ψ0_smpl,NN_edges,Julia_Python_node_mapping)
            nnn_correlations_GNN_edges_dict = NN_NNN_szs_per_samples(g,ψ0_smpl,NNN_edges,Julia_Python_node_mapping)
            magnetization_dict_X = on_site_szs_per_samples(g,ψ0_X_smpl)
            nn_correlations_GNN_edges_dict_X = NN_NNN_szs_per_samples(g,ψ0_X_smpl,NN_edges,Julia_Python_node_mapping)
            nnn_correlations_GNN_edges_dict_X = NN_NNN_szs_per_samples(g,ψ0_X_smpl,NNN_edges,Julia_Python_node_mapping)
        
            Mg = [magnetization_dict[Julia_Python_node_mapping[node]] for node in nodes]
            NN_corrs_GNN = [nn_correlations_GNN_edges_dict[NN_edge] for NN_edge in NN_edges]
            NNN_corrs_GNN = [nnn_correlations_GNN_edges_dict[NNN_edge] for NNN_edge in NNN_edges]
            Mg_X = [magnetization_dict_X[Julia_Python_node_mapping[node]] for node in nodes]
            NN_corrs_GNN_X = [nn_correlations_GNN_edges_dict_X[NN_edge] for NN_edge in NN_edges]
            NNN_corrs_GNN_X = [nnn_correlations_GNN_edges_dict_X[NNN_edge] for NNN_edge in NNN_edges]
        
            npzwrite(folder_name * "/hxs_$(smpl+δ_snapshot+largest_int).npy", hxs)
            npzwrite(folder_name * "/Mg_$(smpl+δ_snapshot+largest_int).npy", Mg)
            npzwrite(folder_name * "/Mg_X_$(smpl+δ_snapshot+largest_int).npy", Mg_X)
            npzwrite(folder_name * "/NN_corrs_$(smpl+δ_snapshot+largest_int).npy", NN_corrs_GNN)
            npzwrite(folder_name * "/NNN_corrs_$(smpl+δ_snapshot+largest_int).npy", NNN_corrs_GNN)
            npzwrite(folder_name * "/NN_corrs_X_$(smpl+δ_snapshot+largest_int).npy", NN_corrs_GNN_X)
            npzwrite(folder_name * "/NNN_corrs_X_$(smpl+δ_snapshot+largest_int).npy", NNN_corrs_GNN_X)

        end
    end      
end

end 