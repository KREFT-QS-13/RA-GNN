include("./Benchmark_exp.jl")
using .Benchmark_exp
using ArgParse

arg = ArgParseSettings()
@add_arg_table! arg begin
    "--nx"
        help = "The number of columns. Default is 4."
        arg_type = Int
        default = 4
    "--ny"
        help = "The number of rows. Default is 4."
        arg_type = Int
        default = 4
    "--alpha"
        help = "The exponent of the power law. Default is 6."
        arg_type = Int
        default = 6
    "--amp_R"
        help = "The amplitude of the power law. Default is 0.0."
        arg_type = Float64
        default = 0.0
    "--folder"
        help = "The folder to save the results. Default is Benchmark."
        arg_type = String
        default = "Benchmark"
end

parsed_args = parse_args(arg)
nx = parsed_args["nx"]
ny = parsed_args["ny"]
alpha = parsed_args["alpha"]
path_to_folder = parsed_args["folder"]

# fix perturabtions such it will be the same for all deltas
R = 10.0; amp_R = parsed_args["amp_R"] 
deltas = range(-50.0, 50.0, step=5.0) # length(deltas) = 22

println("Alpha = $alpha")
println("Size = $nx x $ny")
println("R = $R with perturbation amp_R = $amp_R")
println("-"^20)

experiment_magnetization(deltas, nx, ny, R, amp_R, alpha, path_to_folder)
