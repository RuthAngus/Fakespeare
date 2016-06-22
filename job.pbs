#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=60GB
#PBS -N fakespeare
#PBS -M danfm@nyu.edu
#PBS -j oe
#PBS -o logs
#PBS -e logs

module purge
module load cuda/7.5.18
module load cudnn/7.0v4.0

export PATH="$HOME/miniconda3/bin:$PATH"
export OMP_NUM_THREADS=1

SRCDIR=$HOME/projects/Fakespeare
RUNDIR=$SCRATCH/Fakespeare/results/${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $SRCDIR
THEANO_FLAGS=device=gpu python train.py -o $RUNDIR

