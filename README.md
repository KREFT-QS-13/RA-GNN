# Second Master Project

All work done in this project relates to the following paper: [Learning interactions between Rydberg atoms](https://arxiv.org/abs/2412.12019) by Anna Dawid, Olivier Simard, et al.

Supervisor: **Anna Dawid**

### Cloning and launching code
```
git clone https://github.com/KREFT-QS-13/RA-GNN.git
```

```
julia --project=. -e 'using Pkg; Pkg.add("Statistics"); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'
```


### Benchmark
- Folder *Benchmark* contains the experiments tracking the error and max truncation error vs the bond dimension for different values of $`\delta`$
- Folder *Benchmark-v2* contains the same experiments, but the sweep scheduling of bond dimension is the same as in the original code (it starts with small values, then grows to the set limit.)
- Folder *Benchmark-alpha* contains error and max truncation error vs the bond dimension for different values of $`\delta`$ and different degree of spin-spin interaction coefficient $` J_{ij}\sim R^{-\alpha}`$ 

To-do list:
- [x] Benchmark the dmrg method from ITensors (Benchmark(-v2) folder)
- [x] Generate datasets based on the paper (20 000 + 2000 per size):
    - [x] 4x4 (χDMRG = 80)
    - [x] 5x5 (χDMRG = 90)
    - [x] 6x6 (χDMRG = 100)
    - [x] 7x6 (χDMRG = 110, only test dataset)
    - [x] 7x7 (χDMRG = 120, only test dataset)
    - [x] 8x7 (χDMRG = 130, only test dataset)
    - [x] 8x8 (χDMRG = 140, only test dataset)
    - [x] 9x8 (χDMRG = 150, only test dataset)
    - [x] 9x9 (χDMRG = 160, only test dataset)
- [x] Fixed datasets: 5x5, 6x6, 7x6
- [ ] ~~Calc the $C_3$ based on experimental value from papers~~
- [x] Make plots trunc_err vs bond dim. for different values $`\alpha=`$ 3,2,1 : $`J_{ij}\sim R^{-\alpha}`$
- [x] Start developing GNN-PNA model
- [x] Train GNN and recreate figure 4. from the papaer only for case 5
- [ ] Recrate the figure 4 just for the MLP network at the end of the current architecture -> TODO: Mapping the input to make it compatible
- [x] Make phase diagram dor benchmarks