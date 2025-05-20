# SIMEC and SIMEXP experiments

To reproduce the experiments, first create an environment and install the required packages (requires Python >=3.12):
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade setuptools
python3 -m pip install -r requirements.txt
```

Until the `simec` package is not published on PyPI, install it in editable mode:
```
python3 -m pip install -e simec
```

Run in the repository directory:
```bash
export PYTHONPATH=$(pwd)
```

## Usage

### Set up environment
- Python version: 3.10.5
- Library requirements: (pipenv, conda, venv users) requirements.txt or (poetry users) pyproject.toml
- System requirements: GPU with CUDA enabled

### Train ViT models
```bash
cd experiments/models
python3 train_vit.py --config cifar-config.json --dataset CIFAR10 --batch_size 64 --epochs 100 --lr 0.01 --save_every 25 --device cuda:0 --exp_name cifar10_experiment --base_dir ../..
python3 train_vit.py --config mnist-config.json --dataset MNIST --batch_size 64 --epochs 20 --lr 0.01 --save_every 5 --device cuda:0 --exp_name mnist_experiment --base_dir ../..
```

### Test ViT model
```bash
python3 models/test_vit.py --model_path ../models/mnist_experiment/model_final.pt
python3 models/test_vit.py --model_path ../models/cifar10_experiment/model_final.pt
```
### Convert images and texts in the expected format
```bash
cd data-preprocessing
bash main.sh
```

### Prepare experiments
```bash
    cd experiments
    python3 prepare_experiment.py --default
```

### All explorations and interpretations
```bash
cd sh-scripts
bash all_experiments.sh
```

### For a single exploration
```bash
python3 vit_exploration.py --experiment-path ./experiments_data/cifar-1p0-16-all --out-dir ../res
python3 bert_exploration.py --experiment-path ./experiments_data/winobias-1p0-16-target-word --out-dir ../res
```

### For a single interpretation
```bash
python3 vit_exploration_interpretation.py --experiment-path experiments_data/cifar-1p0-16-all --results-dir ../res/cifar-1p0-16-all-20250403-200221 # adjust to your case
python3 bert_exploration_interpretation.py --experiment-path experiments_data/winobias-1p0-16-target-word --results-dir ../res/winobias-1p0-16-target-word-20250416-075008 # adjust to your case
```

### Group experiments
```bash
python group_results.py --experiment-path ../experiments/experiments_data/ --results-dir ../res --out-name all_experiments
```

### Run dashboard
```bash
cd dashboard
python3 app.py
```

### Result notebook
```bash
cd notebooks
plots_and_tables.ipynb
```

