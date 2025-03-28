"""
Module for benchmarking DMRG calculations 
"""
module Benchmark_exp

include("./DMRG.jl")

using .DMRG

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

export exp_TFI_DMRG, experiment_err_vs_bond_dim

# C6 coefficient and perfect r = 7 um
const C6 = 5420158.53

# ------------------------------------------------------------------------------------------------
#
# CODE FOR EXPERIMENT 1: Error vs Bond Dimension
#
# ------------------------------------------------------------------------------------------------
# To used any futher but ready for more andvanced use -> moved to just DMRGObserver
mutable struct MyObserver <: AbstractObserver
    max_trunc_error::Float64
    truncerr_history::Vector{Float64}
end

# Constructor with default values
MyObserver() = MyObserver(0.0, Float64[])

function ITensors.measure!(obs::MyObserver; kwargs...)
    if haskey(kwargs, :spec)
        spec = kwargs[:spec]
        if !isnothing(spec)
            trunc_err = spec.truncerr
            obs.max_trunc_error = max(obs.max_trunc_error, trunc_err)
            push!(obs.truncerr_history, trunc_err)
        end
    end
end

function exp_TFI_DMRG(g::NamedGraph, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, sites; nsweeps=20, hs::Union{Nothing,Dict} = nothing, Jzzs::Vector{Float64})
    L = length(vertices(g))
    if hs === nothing
        hs = Dict(zip(vertices(g), [1.0 for e in vertices(g)]))
    end

    # Build the Hamiltonian
    H = HTFI(g, hs, Jzzs, edgs, sites)

    # Initial state for DMRG routine (all spins pointing up)
    init_state = ["Up" for _ in 1:L]
    ψ0 = ITensors.randomMPS(sites, init_state; linkdims = 2)

    println("Starting DMRG for reference energy...")
    sweeps = Sweeps(nsweeps)
    cutoff!(sweeps, 1E-18)
    # maxdim!(sweeps, Tuple(min(2^(floor(Int64, 0.5*i)), 200)  for i in 1:nsweeps)...)
    maxdim!(sweeps, 200)

    E_ref, _ = dmrg(H, ψ0, sweeps)
    println("Reference energy (high precision) = ", E_ref)

    maxdims = collect(range(5, 100, step=5))
    pushfirst!(maxdims, 1)  # Add 1 at the beginning

    # Run DMRG Experiment
    errors = Float64[]  # Store error values
    max_truncation_errors = Float64[]  # Store maximum truncation errors
    
    for md in maxdims
        d = hs[first(vertices(g))]
        println("-"^20)
        println("Starting DMRG for delta = $d bond dimension = $md:")

        obs = DMRGObserver() # Reset observer for each bond dimension
        sweeps = Sweeps(nsweeps)
        # maxdim!(sweeps, Tuple(min(2^(floor(Int64, 0.5*i)), md)  for i in 1:nsweeps)...)
        maxdim!(sweeps, md)

        E, _ = dmrg(H, ψ0, sweeps; observer=obs)

        error = abs(E - E_ref)  # Compute absolute error
        push!(errors, error)  # Store the error
        push!(max_truncation_errors, obs.truncerrs[end])
         
        println("maxdim = $md: Energy = $E, Error = $error, Max truncation error = $(obs.truncerrs[end])")
        println("-"^20*"\n")
    end

    println("Finished.")

    return errors, maxdims, max_truncation_errors
end


function experiment_err_vs_bond_dim(nx::Int, ny::Int, R::Float64, amp_R::Float64, hx::Float64, alpha::Int=6, path_to_folder::String="./Experiment_1")
    # Create the directory structure first
    size_dir = joinpath(path_to_folder, "$(nx)x$(ny)")
    mkpath(size_dir)
    
    # Create the file name
    if amp_R != 0.0
        filename = joinpath(size_dir, "data_err_vs_maxdim_$(nx)x$(ny)_delta=$(hx)_amp_R=$(amp_R).npz")
    else
        filename = joinpath(size_dir, "data_err_vs_maxdim_$(nx)x$(ny)_delta=$(hx).npz")
    end

    println("Experiment 1: Error vs Bond Dimension")
    println("Saving results to: $filename")


    # Define the 2D grid as a named graph with vertices (i,j) where i is the row and j is the column
    # NOTE WE ARE IMPLICITING ORDERING THE MPS VERTICES AS A SNAKE THROUGH THE LATTICE. 
    # TO CHANGE THE ORDERING OF THE SITES (WHICH MIGHT HELP/HINDER THE SIMULATION) THEN WE NEED
    # TO REORDER THE LIST vertices(g) APPROPRIATELY
    g = NamedGraphs.NamedGraphGenerators.named_grid((nx, ny))

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
    # @show nn

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

    
    # Get the indices for the MPS. Assume no conservation laws
    sites = siteinds("S=1/2", length(vertices(g)); conserve_qns=false)

    println("The number of available threads is : $(Sys.CPU_THREADS)")
    println("The number of threads used: $(Threads.nthreads())")
    # H_gates = ops([("Ry", n, (θ=π / 2,)) for n in 1:nx*ny], sites)
    
    # Realization:
    perturbations = [(rescaled_rand(-amp_R,amp_R), rescaled_rand(-amp_R,amp_R)) for _ in 1:length(nominal_values)]
    perturbations = reshape(perturbations,size(nominal_values))
    pstns = nominal_values.+perturbations

    # all-to-all distances
    distances = pairwise(
        Euclidean(),
        reshape(pstns,length(pstns)),
        reshape(pstns,length(pstns))
    )

    distances = distances[mask]
    Jij = C6./(distances.^alpha)

    # Simplified version for num_δs = 1
    # Create uniform field strength for all vertices
    hhh = Dict(zip(vertices(g), fill(hx, num_vertices)))
       
    # Run DMRG experiment
    @time errors, maxdims, max_truncation_errors = exp_TFI_DMRG(g, edges_array, sites; hs = hhh, Jzzs = Jij)
    println("Finished DMRG experiment.")
    println("-"^20*"\n")

    # Save results
    npzwrite(filename, Dict(
        "hxs" => fill(hx, num_vertices),
        "errors" => errors,
        "maxdims" => maxdims,
        "max_truncation_errors" => max_truncation_errors
    ))
end

end