include("./Benchmark_exp.jl")
using .Benchmark_exp
using OrderedCollections, NPZ
using JSON
using ArgParse

# Set up argument parsing
arg = ArgParseSettings()
@add_arg_table! arg begin
    "--params"
        help = "Path to the JSON file containing benchmark parameters"
        arg_type = String
        default = "benchmark_parameters.json"
end

# Parse arguments
parsed_args = parse_args(arg)
params_file = parsed_args["params"]

# Load parameters
nx, ny, alpha, R, amp_R, deltas, bond_dims, quick_start, ref_bond_dim, path_to_folder = load_parameters(params_file)

println("Alpha = $alpha")
println("Size = $nx x $ny")
println("R = $R with perturbation amp_R = $amp_R")
println("-"^20)

# create a dictionary to store the results
staggered_magnetization = OrderedDict{Int, OrderedDict{Float64, Float64}}(bd => OrderedDict{Float64, Float64}(md => 0.0 for md in deltas) for bd in bond_dims)
magnetization_per_site = OrderedDict{Int, OrderedDict{Float64, Vector{Float64}}}(bd => OrderedDict{Float64, Vector{Float64}}(md => Float64[] for md in deltas) for bd in bond_dims)

println("Setting up lattice for all experiments:")
lattice_params = Benchmark_exp.setup_lattice(nx, ny, R, amp_R, alpha, path_to_folder)
println("Lattice setup complete")
println("Lattice ptns: $(lattice_params[end])")
println("-"^20)
for d in deltas
    println("Running experiment for delta = $d")
    mag_per_site, mag_per_bond_dim = Benchmark_exp.experiment_err_vs_bond_dim(nx, ny, lattice_params, d, bond_dims, alpha, quick_start, ref_bond_dim, path_to_folder)
    
    for bd in bond_dims
        staggered_magnetization[bd][d] = mag_per_bond_dim[bd]
        magnetization_per_site[bd][d] = mag_per_site[bd]
    end
    println("Done for delta = $d")
    # println("Magnetization per site:\n $mag_per_site")
    println("Staggered magnetization:\n $staggered_magnetization")
    println("-"^20)
end

Benchmark_exp.save_dict_int_to_pairs(path_to_folder * "/$nx" * "x" * "$ny" * "/staggered_magnetization.npz", staggered_magnetization)
# TODO: Fix magnetization_per_site so it will work as staggered_magnetization and be saved
# Benchmark_exp.save_dict_int_to_pairs(path_to_folder * "/magnetization_per_site.npz", magnetization_per_site)
