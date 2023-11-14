module load python
module load openslide
module load scipy-stack
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
python -m pip install --no-index -r requirements.txt