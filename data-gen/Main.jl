include("./DMRG.jl")

using .DMRG

# nx is num columns, ny is num rows
nx, ny = 4,4
# number of samples to realize (number of distinct Jzz configurations, given that each of those imply num_δs constant transverse fields)
num_realization = 2000 # the effective total number of realizations is num_realization*num_δs
num_δs = 10  # If one uses a single delta, `amp_delta' has be set to null

# Select the way to calculate correlation functions
is_sampled = false
num_samples = 1000

# Maximum bond dimension for ground state and time evolution 
# (accuracy parameter, χ -> infinity is guaranteed to be exact. Simulation time scales as χ^{3})
χDMRG  = 80

# Initial x-field strength for the ground state simulation
R = 10.0; amp_R = 0.2  # in micrometers 
delta = 0.0; amp_delta = 100.0 # omega in the papaer

path_to_folder = "./Datasets/dataset_NO_Dr_X_Mg_NN_NNN_delta_one" #"./Mg_NN_NNN_delta_hist/dataset_mps_NNN_CTN_RYD_ALL"
if is_sampled
    path_to_folder = path_to_folder * "_SMPL_$(num_samples)"  #"./Mg_NN_NNN_delta_hist/dataset_mps_NNN_CTN_RYD_ALL/"
end

main_Mg_NN_NNN_δ(nx,ny,
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
