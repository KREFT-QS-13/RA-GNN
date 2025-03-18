include("./Benchmark_exp.jl")
using .Benchmark_exp

nx, ny = 4,4

# fix perturabtions such it will be the same for all deltas
R = 10.0; amp_R = 1.0 
deltas = [0.0, 10.0, 20.0, 25.0, 30.0, 50.0, 100.0]

for d in deltas
    experiment_err_vs_bond_dim(nx, ny, R, amp_R, d)
end