include("./DMRG.jl")

using .DMRG

nx, ny = 4,5

R = 10.0; amp_R = 0.0 
deltas = [0.0, 10.0, 20.0, 25.0, 30.0, 50.0, 100.0]

for d in deltas
    experiment_err_vs_bond_dim(nx, ny, R, amp_R, d)
end