#!/bin/bash 
#SBATCH --job-name=demo1

#SBATCH -p v100
#SBATCH --gres=gpu:v100:1


#SBATCH --output experiment-%j-resutls.output # -o
#SBATCH --error experiment-%j-errorlog.output # -e


echo "I ran on node: "
hostname
sleep 20

