include("./DMRG.jl")

using .DMRG
using ArgParse

arg = ArgParseSettings()
@add_arg_table! arg begin
    "--params"
        help = "Path to the JSON file containing data generation parameters"
        arg_type = String
        default = "data-gen/params/data_gen_params_a=2_4x4.json"
end

# Parse arguments
parsed_args = parse_args(arg)
params_file = parsed_args["params"]

hamiltonian_params, generation_params, save_folder, is_test_dataset = load_params_from_json(params_file)
# nx is num columns, ny is num rows
nx, ny = hamiltonian_params["nx"], hamiltonian_params["ny"]
R = hamiltonian_params["R"]; amp_R = hamiltonian_params["amp_R"]  # in micrometers 
alpha = hamiltonian_params["alpha"]

# number of samples to realize (number of distinct Jzz configurations, given that each of those imply num_δs constant transverse fields)
num_realization = generation_params["num_realization"] # the effective total number of realizations is num_realization*num_δs
num_δs = generation_params["num_δs"]  # If one uses a single delta, `amp_delta' has be set to null

# Select the way to calculate correlation functions
is_sampled = generation_params["is_sampled"]
num_samples = generation_params["num_samples"]

# Maximum bond dimension for ground state and time evolution 
# (accuracy parameter, χ -> infinity is guaranteed to be exact. Simulation time scales as χ^{3})
χDMRG  = generation_params["χDMRG"] # increase by 10 for different sizes nx, ny = 5,5 -> χDMRG = 90

# Initial x-field strength for the ground state simulation
delta = generation_params["delta"]; amp_delta = generation_params["amp_delta"] # omega in the papaer

path_to_folder = save_folder * "/alpha=$(alpha)"
# "./Datasets/dataset_NO_Dr_X_Mg_NN_NNN_delta_one" #"./Mg_NN_NNN_delta_hist/dataset_mps_NNN_CTN_RYD_ALL"
if is_sampled
    path_to_folder = path_to_folder * "_SMPL_$(num_samples)"  #"./Mg_NN_NNN_delta_hist/dataset_mps_NNN_CTN_RYD_ALL/"
end

if is_test_dataset
    path_to_folder = path_to_folder  * "/test"
    mkpath(path_to_folder)
end

println("Starting data generation...")
println("Hamiltonian parameters:")
println("nx = $nx, ny = $ny, alpha = $alpha, R = $R, amp_R = $amp_R")
println("Generation parameters:")
println("num_realization = $num_realization, num_δs = $num_δs, χDMRG = $χDMRG, delta = $delta, amp_delta = $amp_delta")
println("-"^20)
start_time = time()
main_Mg_NN_NNN_δ(nx,ny,
                alpha,
                num_realization,
                num_δs,
                χDMRG,
                R,
                amp_R,
                delta,
                amp_delta,
                path_to_folder;
                is_sampled=is_sampled,
                num_samples=num_samples)

end_time = time()
println("-"^20)
println("Data generation completed in $(end_time - start_time) seconds.")
println("Average time per realization: $((end_time - start_time) / num_realization) seconds.")
println("Average time per δ: $((end_time - start_time) / num_realization / num_δs) seconds.")
println("Data saved to $path_to_folder")