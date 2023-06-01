#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=3000M
#SBATCH --time=0-00:30
#SBATCH --output=tf-%j.out

module load python
virtualenv --no-download tensorflow

module load cuda cudnn
module load scipy-stack
source tensorflow/bin/activate
pip install --no-index tensorflow
python ../src/transfer_learning.py 10_20.jpeg features.csv
deactivate
rm -rf tensorflow/bin/activate