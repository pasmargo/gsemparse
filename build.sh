#!/bin/bash

conda create --yes -n py3n python=3
echo "source activate py3n" > python_env.sh
chmod u+x python_env.sh
source python_env.sh
conda install -c anaconda tensorflow-gpu keras-gpu scikit-learn nomkl
conda install pillow matplotlib joblib
conda install -c conda-forge pudb
pip install opencv-python
