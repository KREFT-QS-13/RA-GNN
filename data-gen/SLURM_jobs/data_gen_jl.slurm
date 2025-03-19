#SBATCH --job-name=data-gen
#SBATCH --output=dg-%x_%j.out
#SBATCH --mail-user="s3994732@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --partition="<partition>"
#SBATCH --time=00:00:15
#SBATCH --ntasks=1
#SBATCH --mem=10M


echo "#### Data generation"
# get the current working directory
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and I am currently in $CWD"

module load Julia/1.10.0-linux-x86_64

echo "#### Finished Test. Have a nice day"
