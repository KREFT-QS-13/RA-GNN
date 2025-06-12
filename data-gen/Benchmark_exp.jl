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
using DataStructures
using JSON

using .Filul
using .Latul

export exp_TFI_DMRG, experiment_err_vs_bond_dim, experiment_magnetization, load_parameters, setup_lattice

# C6 coefficient and perfect r = 7 um
# const C6 = 5420158.53
# ^^ NOT IN USE ANYMORE, left for reference

# ------------------------------------------------------------------------------------------------
#
# Parameter Loading Functions
#
# ------------------------------------------------------------------------------------------------
function load_parameters(json_file::String)
    """
    Load benchmark parameters from a JSON file.
    
    Parameters:
    -----------
    json_file : String
        Path to the JSON file containing benchmark parameters
        
    Returns:
    --------
    Tuple containing:
    - nx, ny : Int
        Lattice dimensions
    - alpha : Int
        Power law exponent
    - R : Float64
        Lattice spacing
    - amp_R : Float64
        Perturbation amplitude
    - C6 : Float64
        Interaction constant
    - init_state : String
        Initial state
    - deltas : Vector{Float64}
        Array of delta values
    - bond_dims : Vector{Int}
        Array of bond dimensions
    - quick_start : Bool
        Whether to use quick start for DMRG
    - ref_bond_dim : Int
        Reference bond dimension
    - path_to_folder : String
        Output folder path
    """
    params = JSON.parsefile(json_file)
    
    # Extract lattice parameters
    lattice = params["lattice"]
    nx = lattice["nx"]
    ny = lattice["ny"]
    alpha = lattice["alpha"]
    R = lattice["R"]
    amp_R = lattice["amp_R"]
    
    # Extract physics parameters
    C6 = params["physics"]["C6"]
    init_state = params["physics"]["init_state"]
    
    # Extract deltas
    deltas = params["deltas"]
    
    # Extract bond dimensions
    bond_params = params["bond_dims"]
    bond_dims = collect(range(bond_params["start"], bond_params["stop"], step=bond_params["step"]))
    if bond_params["include_1"]
        pushfirst!(bond_dims, 1)
    end
    
    # Extract DMRG parameters
    dmrg_params = params["dmrg"]
    quick_start = dmrg_params["quick_start"]
    ref_bond_dim = dmrg_params["ref_bond_dim"]
    
    # Extract output parameters
    path_to_folder = params["output"]["folder"]
    
    return nx, ny, alpha, R, amp_R, C6, init_state, deltas, bond_dims, quick_start, ref_bond_dim, path_to_folder
end

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


function calculate_staggered_magnetization(mag_per_site::Vector{Float64}, L::Int)
    """
    Calculate the staggered (Neel) magnetization from the magnetization per site.
    
    Parameters:
    -----------
    mag_per_site : Vector{Float64}
        Vector of magnetization values for each site
    L : Int
        Linear dimension of the lattice (LxL)
    
    Returns:
    --------
    Float64
        Absolute value of the staggered magnetization
    """
    # Create the staggered pattern matrix
    stagger = [(-1)^(i + j) for i in 1:L, j in 1:L]
    
    # Reshape the magnetization array to LxL and calculate the staggered magnetization
    mag_matrix = reshape(mag_per_site, L, L)
    magnetization = abs(mean(mag_matrix .* stagger))
    
    return magnetization
end

function save_dict_int_to_pairs(filename::String, dict::OrderedDict{Int, OrderedDict{T, V}}) where {T <: Union{Int, Float64}, V}
    # Convert the dictionary into arrays
    outer_keys = collect(keys(dict))
    inner_keys = collect(keys(first(values(dict))))
    
    # Create arrays to store the values
    if V <: Vector
        # For magnetization_per_site which has Vector{Float64} values
        values_array = Array{Array{Float64, 2}, 1}(undef, length(outer_keys))
        for (i, outer_key) in enumerate(outer_keys)
            inner_dict = dict[outer_key]
            # Convert each inner dictionary's values to a 2D array
            values_array[i] = hcat([inner_dict[k] for k in inner_keys]...)
        end
    else
        # For staggered_magnetization which has Float64 values
        values_array = Array{Float64, 2}(undef, length(outer_keys), length(inner_keys))
        for (i, outer_key) in enumerate(outer_keys)
            inner_dict = dict[outer_key]
            values_array[i, :] = [inner_dict[k] for k in inner_keys]
        end
    end
    
    # Save everything to a .npz file
    npzwrite(filename, Dict(
        "outer_keys" => outer_keys,
        "inner_keys" => inner_keys,
        "values" => values_array
    ))
end

function get_output_path(base_path::String, alpha::Int, nx::Int, ny::Int)::String
    """
    Create and return the output path with alpha and size folders.
    
    Args:
        base_path: Base output path from JSON
        alpha: Alpha value for the folder name
        nx, ny: Lattice dimensions for the size folder
        
    Returns:
        Full path including alpha and size folders
    """
    alpha_path = joinpath(base_path, "alpha_$(alpha)")
    size_path = joinpath(alpha_path, "$(nx)x$(ny)")
    mkpath(size_path)
    return size_path
end

function exp_TFI_DMRG(g::NamedGraph, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, 
                      sites; nsweeps=20, hs::Union{Nothing,Dict} = nothing, Jzzs::Vector{Float64}, 
                      init_state::String, bonds_dims::Vector{Int}=collect(range(5, 100, step=5)), ref_bond_dim::Int=200, quick_start::Bool=true)
    L = length(vertices(g))
    size_L = trunc(Int, sqrt(L))
    if hs === nothing
        hs = Dict(zip(vertices(g), [1.0 for e in vertices(g)]))
    end

    nodes = [i for i=0:length(vertices(g))-1]
    Julia_Python_node_mapping = Dict(zip(nodes, vertices(g)))

    # Build the Hamiltonian
    H = HTFI(g, hs, Jzzs, edgs, sites)

    # Initial state for DMRG routine (all spins pointing up)

    if init_state == "FM"
        init_state = ["Up" for _ in 1:L]
    elseif init_state == "AFM"
        init_state = [i%2 == 0 ? "Up" : "Dn" for i in 1:L]
    else
        error("Invalid initial state: $init_state. Only FM and AFM are supported.")
    end
    println("Initial state: $(init_state)")
    
    println("Starting DMRG for reference energy...")
    sweeps = Sweeps(nsweeps)
    cutoff!(sweeps, 1E-18)
    if quick_start
        maxdim!(sweeps, Tuple(min(2^(floor(Int64, 0.5*i)), ref_bond_dim)  for i in 1:nsweeps)...)
    else
        maxdim!(sweeps, ref_bond_dim)
    end
    
    ψ0 = ITensors.randomMPS(sites, init_state; linkdims = 2)
    E_ref, _ = dmrg(H, ψ0, sweeps)
    println("Reference energy (high precision) = ", E_ref)

    # Run DMRG Experiment
    errors = Float64[]  # Store error values
    max_truncation_errors = Float64[]  # Store maximum truncation errors

    list_magnetization_per_bond_dim = OrderedDict(md=>0.0 for md in bonds_dims) 
    list_magnetization_per_site = OrderedDict(md=>Float64[] for md in bonds_dims)  

    for md in bonds_dims
        delta = hs[first(vertices(g))]
        println("-"^20)
        println("Starting DMRG for delta = $delta bond dimension = $md:")

        obs = DMRGObserver() # Reset observer for each bond dimension
        sweeps = Sweeps(nsweeps)
        if quick_start
            maxdim!(sweeps, Tuple(min(2^(floor(Int64, 0.5*i)), md)  for i in 1:nsweeps)...)
        else
            maxdim!(sweeps, md)
        end

        ψ0 = ITensors.randomMPS(sites, init_state; linkdims = 2)
        E, ψ0 = dmrg(H, ψ0, sweeps; observer=obs)

        error = abs(E - E_ref)  # Compute absolute error
        push!(errors, error)  # Store the error
        push!(max_truncation_errors, obs.truncerrs[end])
         
        println("maxdim = $md: Energy = $E, Error = $error, Max truncation error = $(obs.truncerrs[end])")
        println("-"^20*"\n")

        # Calculate magnetization for this bond dimension
        magnetization_dict = Dict(zip(vertices(g), expect(ψ0, "Z")))
        Mg = [magnetization_dict[Julia_Python_node_mapping[node]] for node in nodes]
        
        staggered_magnetization = calculate_staggered_magnetization(Mg, size_L)

        println("Magnetization per site: $Mg")
        println("Staggered magnetization: $staggered_magnetization")

        list_magnetization_per_site[md] = copy(Mg)
        list_magnetization_per_bond_dim[md] = staggered_magnetization

    end

    println("Finished.")
    return errors, bonds_dims, max_truncation_errors, list_magnetization_per_site, list_magnetization_per_bond_dim
end

function exp2_TFI_DMRG(g::NamedGraph, edgs::Vector{Tuple{Tuple{Int64, Int64},Tuple{Int64,Int64}}}, sites; nsweeps=20, hs::Union{Nothing,Dict} = nothing, Jzzs::Vector{Float64})
    L = length(vertices(g))
    size_L = trunc(Int, sqrt(L))
    if hs === nothing
        hs = Dict(zip(vertices(g), [1.0 for e in vertices(g)]))
    end

    nodes = [i for i=0:length(vertices(g))-1]
    Julia_Python_node_mapping = Dict(zip(nodes, vertices(g)))

    # Build the Hamiltonian
    H = HTFI(g, hs, Jzzs, edgs, sites)

    # Initial state for DMRG routine (all spins pointing up)
    init_state = ["Up" for _ in 1:L]
    ψ0 = ITensors.randomMPS(sites, init_state; linkdims = 2)

    println("Starting DMRG for magnetization...")
    # maxdims = collect(range(5, 100, step=5))
    # pushfirst!(maxdims, 1)  # Add 1 at the beginning
    # maxdims = [1,5,50,100]
    maxdims = 150

   
    delta = hs[first(vertices(g))]
    println("-"^20)
    println("Starting DMRG for delta = $delta bond dimension = $maxdims:")

    sweeps = Sweeps(nsweeps)
    cutoff!(sweeps, 1E-18)
    maxdim!(sweeps, Tuple(min(2^(floor(Int64, 0.5*i)), maxdims)  for i in 1:nsweeps)...)
    # maxdim!(sweeps, maxdims)
    
    _, ψ0 = dmrg(H, ψ0, sweeps)

    # Calculate magnetization for this bond dimension
    magnetization_dict = Dict(zip(vertices(g), expect(ψ0, "Z")))
    Mg_per_site = [magnetization_dict[Julia_Python_node_mapping[node]] for node in nodes]
    Mg = calculate_staggered_magnetization(Mg_per_site, size_L)
    println("For bond dimension $maxdims, the staggered magnetization is $Mg")
    println("-"^20*"\n")


    println("Finished.")
    return Mg
end


function setup_lattice(nx::Int, ny::Int, R::Float64, amp_R::Float64, C6::Float64, init_state::String="FM", alpha::Int=6, path_to_folder::String="./Experiment_1")
    # Remove redundant path creation since it's done in main_benchmark.jl
    # path_to_folder = get_output_path(path_to_folder, alpha, nx, ny)
    
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

    npzwrite(joinpath(path_to_folder, "lattice_$(nx)x$(ny)_R=$(R)_pm_$(amp_R)_init=$(init_state).npz"), Dict(
        "alpha" => alpha,
        "R" => R,
        "amp_R" => amp_R,
        "C6" => C6,
        "Jij" => Jij,
        "distances" => distances,
    ))

    return R, amp_R, g, edges_array, sites, Jij, num_vertices, pstns
end


function experiment_err_vs_bond_dim(nx::Int, ny::Int, lattice_params, hx::Float64, init_state::String, bonds_dims::Vector{Int}, alpha::Int=6, quick_start::Bool=true, ref_bond_dim::Int=200, path_to_folder::String="./Experiment_1")
    # Remove redundant path creation since it's done in main_benchmark.jl
    # path_to_folder = get_output_path(path_to_folder, alpha, nx, ny)
    
    _, amp_R, g, edges_array, sites, Jij, num_vertices, _ = lattice_params

    # Create the file name with init_state included
    if amp_R != 0.0
        filename = joinpath(path_to_folder, "data_err_vs_maxdim_delta=$(hx)_amp_R=$(amp_R)_init=$(init_state).npz")
        filename_mag = joinpath(path_to_folder, "Mg_init=$(init_state)_delta=$(hx)_amp_R=$(amp_R).npz")
    else
        filename = joinpath(path_to_folder, "data_err_vs_maxdim_delta=$(hx)_init=$(init_state).npz")
        filename_mag = joinpath(path_to_folder, "Mg_init=$(init_state)_delta=$(hx).npz")
    end

    println("Experiment 1: Error vs Bond Dimension")
    println("Saving results to: $filename")



    # Simplified version for num_δs = 1
    # Create uniform field strength for all vertices
    # hxs = ones(num_vertices).*hxs_grid[δ_snapshot+1] #ones(length(edges(g)))*0
    # hhh = Dict(zip(vertices(g), hxs))
    hhh = Dict(zip(vertices(g), fill(hx, num_vertices)))
       
    # Run DMRG experiment
    @time results = exp_TFI_DMRG(g, 
                                edges_array, 
                                sites; 
                                hs = hhh, 
                                Jzzs = Jij, 
                                init_state=init_state,
                                bonds_dims=bonds_dims, 
                                quick_start=quick_start,
                                ref_bond_dim=ref_bond_dim)
                                
    errors, bonds_dims, max_truncation_errors, list_magnetization_per_site, list_magnetization_per_bond_dim = results
    
    println("Finished DMRG experiment.")
    println("-"^20*"\n")

    # Save the main data file
    npzwrite(filename, Dict(
        "hxs" => fill(hx, num_vertices),
        "errors" => errors,
        "maxdims" => bonds_dims,
        "max_truncation_errors" => max_truncation_errors,
    ))


    return list_magnetization_per_site, list_magnetization_per_bond_dim
end


function experiment_magnetization(deltas, nx::Int, ny::Int, R::Float64, amp_R::Float64, alpha::Int=6, path_to_folder::String="./Experiment_1", max_bond_dim::Int=200, init_state::String="FM")
    # Remove redundant path creation since it's done in main_benchmark.jl
    # path_to_folder = get_output_path(path_to_folder, alpha, nx, ny)
    
    # Initialize dictionary with Float64 values instead of arrays
    magnetization = OrderedDict(delta=>0.0 for delta in sort(deltas))
    println("Initial magnetization dictionary: $magnetization")
    
    for hx in deltas       
        println("Experiment 2: Magnetization - Phase Transition")
        
        
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
            # hxs = ones(num_vertices).*hxs_grid[δ_snapshot+1] #ones(length(edges(g)))*0
            # hhh = Dict(zip(vertices(g), hxs))
            hhh = Dict(zip(vertices(g), fill(hx, num_vertices)))
            
            # Run DMRG experiment
            
            @time mag = exp2_TFI_DMRG(g, edges_array, sites; hs = hhh, Jzzs = Jij)
            magnetization[hx] = mag  # Now this assignment will work
            println("Finished DMRG experiment.\n")
            println("Magnetization to plot: $magnetization")
            println("-"^20*"\n")
        end

    deltas_min, deltas_max = minimum(deltas), maximum(deltas)
    deltas_size =  length(deltas)
    if amp_R != 0.0
        filename_mag = joinpath(path_to_folder, "Mg_init=$(init_state)_delta_$(deltas_min)_$(deltas_max)_$(deltas_size)_amp_R=$(amp_R).npz")
    else
        filename_mag = joinpath(path_to_folder, "Mg_init=$(init_state)_delta_$(deltas_min)_$(deltas_max)_$(deltas_size).npz")
    end
    println("Saving results to: $filename_mag")

    # Convert OrderedDict to separate arrays for keys and values
    delta_values = collect(keys(magnetization))
    mag_values = collect(values(magnetization))
    
    # Save as a dictionary with separate arrays
    npzwrite(filename_mag, Dict(
        "deltas" => delta_values,
        "magnetization" => mag_values
    ))
end

end