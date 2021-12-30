#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=dynamic_job2
#SBATCH --output=/home/oehlers/lowbuffer-%j.out
#SBATCH --error=/home/oehlers/lowbuffer-%j.err
#SBATCH --ntasks=101

module load anaconda/2.3.0
source activate /p/projects/synchronet/python

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so


dir=/home/oehlers/all
#cd $dir

#mkdir savedata
#cd savedata

## srun version
srun --mpi=pmi2 -n $SLURM_NTASKS python $dir/lowbuffer.py

mv /home/oehlers/lowbuffer-$SLURM_JOBID.out ~/outerror
mv /home/oehlers/lowbuffer-$SLURM_JOBID.err ~/outerror


#new=$( ls | grep heatmap )
#mv $new ../
#newname=${new::-4}
#cd ../
#mv savedata $newname
#rm $newname/$new
#mv $newname /home/oehlers
#cd 
#mv $test deleted

