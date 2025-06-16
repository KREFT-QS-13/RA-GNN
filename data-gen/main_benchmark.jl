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

start_time = time()
# Load parameters
nx, ny, alpha, R, amp_R, C6, init_state, deltas, bond_dims, quick_start, ref_bond_dim, init_linkdims, path_to_folder = load_parameters(params_file)

# Get the output path with alpha and size folders
path_to_folder = Benchmark_exp.get_output_path(path_to_folder, alpha, nx, ny)

println("Alpha = $alpha")
println("Size = $nx x $ny")
println("R = $R with perturbation amp_R = $amp_R")
println("C6 = $C6")
println("Initial bond dimensions = $init_linkdims")
println("Output folder: $path_to_folder")
println("-"^20)

# create a dictionary to store the results
staggered_magnetization = OrderedDict{Int, OrderedDict{Float64, Float64}}(bd => OrderedDict{Float64, Float64}(d => 0.0 for d in deltas) for bd in bond_dims)
magnetization_per_site = OrderedDict{Int, OrderedDict{Float64, Vector{Float64}}}(bd => OrderedDict{Float64, Vector{Float64}}(d => Float64[] for d in deltas) for bd in bond_dims)
drmg_time_list = OrderedDict{Int, OrderedDict{Float64, Float64}}(bd => OrderedDict{Float64, Float64}(d => 0.0 for d in deltas) for bd in [bond_dims; -1.0])

println("Setting up lattice for all experiments:")
lattice_params = Benchmark_exp.setup_lattice(nx, ny, R, amp_R, C6, init_state, init_linkdims, alpha, path_to_folder)
println("Lattice setup complete")
println("Lattice ptns: $(lattice_params[end])")
println("-"^20)
for d in deltas
    println("Running experiment for delta = $d")
    mag_per_site, mag_per_bond_dim, drmg_time = Benchmark_exp.experiment_err_vs_bond_dim(nx, ny, lattice_params, d, init_state, bond_dims, alpha, quick_start, ref_bond_dim, init_linkdims, path_to_folder)
    
    for bd in bond_dims
        staggered_magnetization[bd][d] = mag_per_bond_dim[bd]
        magnetization_per_site[bd][d] = mag_per_site[bd]
        drmg_time_list[bd][d] = drmg_time[bd]
    end
    println("Done for delta = $d")
    # println("Magnetization per site:\n $mag_per_site")
    println("Staggered magnetization:\n $staggered_magnetization")
    println("-"^20)
end
println("Outer keys: $(keys(staggered_magnetization))")
println("Inner keys: $(keys(staggered_magnetization[1]))")

# Save the staggered magnetization data
filename = joinpath(path_to_folder, "staggered_magnetization_$(nx)x$(ny)_alpha=$(alpha)_R=$(R)_amp_R=$(amp_R)_init=$(init_state)_initdim=$(init_linkdims).npz")
Benchmark_exp.save_dict_int_to_pairs(filename, staggered_magnetization)
println("Saved staggered magnetization to: $filename")

total_time = time() - start_time
println("Total time of experiment: $total_time seconds")
drmg_time_list[-1.0][0.0] = total_time
# Save the time taken data
filename = joinpath(path_to_folder, "drmg_time_$(nx)x$(ny)_alpha=$(alpha)_R=$(R)_amp_R=$(amp_R)_init=$(init_state)_initdim=$(init_linkdims).npz")
Benchmark_exp.save_dict_int_to_pairs(filename, drmg_time_list)
println("Saved time taken to: $filename")


# TODO: Fix magnetization_per_site so it will work as staggered_magnetization and be saved
# Benchmark_exp.save_dict_int_to_pairs(path_to_folder * "/magnetization_per_site.npz", magnetization_per_site)