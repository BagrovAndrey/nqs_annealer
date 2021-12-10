#!/bin/bash
#SBATCH -A snic2021-1-36
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=5-00:00:00
#SBATCH --gpus-per-task=1

. /software/sse/easybuild/prefix/software/Anaconda3/2020.07-extras-nsc1/lib/python3.8/site-packages/conda/shell/etc/profile.d/conda.sh

conda activate annealing

echo "hello world"
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

python3 -c "import annealing_sign_problem.train; annealing_sign_problem.train.run_triangle_6x6()" --seed 136 --widths '[40, 40, 40]' --kernels 4 --samples 150000 --epochs 300 --batch-size 256 --iters 500 --lr 1.4e-2 --momentum 0.7
