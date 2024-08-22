#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --account=def-haricots

helptext="
    This is a script to perform nested cross validation on a chosen type of ensemble.
    The results will be stored in the sub-directory ./m/ where m is the method of choice.
   
    Usage: `basename $0` [-h] [-m, --method method] [-s, --src path] [-d, --dest path]
    Where:
           [-h]                       show this help text
           [-m, --method method]      select an ensemble method. Pick between l1o, bagging or mlp (default: l1o)
           [-s, --src path]           provide the path of the dataset to be used (default: CDBN/fullDatasetSY.csv)
           [-d, --dest path]          provide the sub-directory where output files should be stored (default: {ensemble_cv})
"

# Default values
method="bagging"
destination="yield_results"
datapath="CDBN/fullDatasetSY.csv"

# Process command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) echo "${helptext}"; exit 0;;
        -d|--dest) destination=$2; shift;;
        -m|--method) method=$2; shift;;
        -s|--src) datapath=$2; shift;;
        *) echo "Unknown parameter passed: $1. Use flags -h or --help for helptext"; exit 1 ;;
    esac
    shift
done

# Load required packages
module load python scipy-stack
source ../venv/bin/activate

# Submit jobs
for i in $(seq 0 25)
do
  sbatch --time 2:30:00 --mem 160G --cpus-per-task 32 --account def-haricots -J "${method}${i}" --wrap "python ensemble.py ${method} ${datapath} ${destination} ${i}"
done
