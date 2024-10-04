#!/bin/bash        

# script for V100
#SBATCH --job-name=test_transformer     # Job name
#SBATCH --partition=v100                # Partition name
#SBATCH --gres=gpu:v100:1                    # Request one GPU
#SBATCH --mem=20G                       # Memory total in MB (for all cores)
#SBATCH --time=0:01:00                 # Time limit hrs:min:sec
#SBATCH --output=./job_log/test_transformer_%j.log  # Standard output and error log
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=jia00129@umn.edu 


# cd ~/
module load gcc/8.2.0 python3/3.10.9_anaconda2023.03_libmamba cuda/11.8.0-gcc-7.2.0-xqzqlf2
# conda activate pytorch-env
which python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# mpirun -np 8 program_name < inputfile > outputfile
# pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

# pip install bitsandbytes-cuda113
cd ~/intro_tutorial
python train_gpu.py
# python3 -m torch.distributed.launch --nproc_per_node=4 temp.py
