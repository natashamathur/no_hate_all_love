#!/bin/bash
module load cuda/10.0
module load midway2
module load python/3.7.0
# python3 -m venv .aml
# virtualenv --system-site-packages .aml
# source .aml/bin/activate

pip install --user nltk
pip install --user sklearn
pip install --user torch
pip install --user numpy
pip install --user pandas


_now="$(date +'%m%d%y_%H%M')"
_outfile="exec_lstm_output_$_now.txt"
python3 jigsaw_toxic/exec_lstm.py --infile train.csv >> $_outfile && echo "Done with exec_lstm.py"
# python3 jigsaw_toxic/exec_lstm.py --infile jigsaw_toxic/ >> $_outfile && echo "Done with exec_lstm.py"

# deactivate

tar -zcvf lstm_results.tar.gz jigsaw_toxic

# call by this to make executable and run:
# chmod u+x rcc_run_model.sh
# sbatch -p gpu2 --time=12:00:00 --account=capp30255 --exclusive --gres=gpu:1 rcc_run_model.sh
