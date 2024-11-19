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

## Experiment preparation
Below is an example of how to use the script `experiments/prepare_experiment.py` to generate an experiment configuration from the command line.

### Command Line Usage Example
```bash

python experiments/prepare_experiments.py \
    -e my_experiment \
    -i 100 \
    -n 10000 \
    -d 5 \
    -t 0.001 \
    -s 10 \
    -r 3 \
    -p all \
    -od ./data/inputs \
    -mp ./models/my_model \
    -o cls \
    -ed ./experiments/results

```
This command:

- Creates an experiment named `my_experiment` in the directory `./experiments/results`.
- Samples 100 inputs from `./data/inputs`.
- Runs 10,000 iterations with a delta multiplier of 5 and a threshold of 0.001.
- Saves results every 10 iterations.
- Repeats the experiment 3 times for each input.
- Explores all patches.
- Uses the model located in `./models/my_model`.
- Assumes the model task is classification (`cls`).

### Default Configuration Example
To generate all predefined experiments (e.g., for MNIST, CIFAR, etc.):
```bash
python experiments/prepare_experiments.py --default
```
This automatically generates experiments for all datasets and configurations defined in the script’s `EXPERIMENTS` dictionary.

#### Experiment Configuration Details
The `EXPERIMENTS` dictionary in the script specifies the dataset-specific configurations for various experiments. These configurations are combined with a `BASE_EXPERIMENT` template to produce a range of experiments.

**`BASE_EXPERIMENT` Template**

This is the default structure applied to all experiments unless overridden by a dataset-specific entry in the `EXPERIMENTS` dictionary:

```python
BASE_EXPERIMENT = {
    "algo": "both",          # Algorithms to run: "simec", "simexp", or both.
    "iterations": 20000,     # Number of iterations to run.
    "delta_mult": None,      # Multiplier for delta adjustment (set dynamically).
    "threshold": 0.01,       # Threshold value for stopping criteria.
    "save_each": 1,          # Frequency of saving results.
    "inputs": None,          # Number of inputs (set dynamically).
    "repeat": None,          # Repetition per input (set dynamically).
    "patches": None,         # Patches to explore (set dynamically or overridden).
    "objective": "cls",      # Task type: "cls" (classification) or "mlm" (masked language model).
}
```

**`EXPERIMENTS` Dictionary**

This dictionary defines configurations specific to each dataset. These settings override or extend the `BASE_EXPERIMENT` template.

```python
EXPERIMENTS = {
    "mnist": {
        "exp_name": "mnist",                   # Name of the experiment.
        "orig_data_dir": "../data/mnist_imgs", # Input data directory.
        "model_path": "../models/vit",         # Model directory or name.
    },
    "cifar": {
        "exp_name": "cifar",
        "orig_data_dir": "../data/cifar10_imgs",
        "model_path": "../models/cifarTrain",
    },
    "hatespeech": {
        "exp_name": "hatespeech",
        "orig_data_dir": "../data/measuring-hate-speech_txts",
        "model_path": "ctoraman/hate-speech-bert",
        "vocab_tokens": None,  # Vocabulary token exploration to be set dynamically.
    },
    "winobias": {
        "exp_name": "winobias",
        "orig_data_dir": "../data/wino_bias_txts",
        "model_path": "bert-base-uncased",
        "patches": "target-word",  # Explores specific target words in MLM tasks.
        "objective": "mlm",        # Task type is MLM for this experiment.
        "vocab_tokens": None,      # Vocabulary token exploration to be set dynamically.
    },
}
```

**Experiments Variations**

The script generates multiple experiments based on the selected datasets by varying parameters dynamically. These parameters include:

- Delta Multiplier (delta_mult):
    - Values: [1, 5].
- Number of Inputs (inputs):
    - For test mode: [10, 2].
    - For full mode: [200, 20].
- Patches (patches):
    - Options: ["all", "one", "q1", "q2", "q3"].
    - If explicitly defined (e.g., target-word for Winobias), only that value is used.
- Vocabulary Tokens (vocab_tokens):
    - For textual experiments: [1, 5, 10].
    - Disabled (None) for non-textual datasets.

**Produced Experiments**

Each combination of parameters and datasets produces a unique experiment configuration. Below are some examples of how the variations are applied:

*Example: MNIST*

For the MNIST dataset:

- Input directory: `../data/mnist_imgs`
- Model: `../models/vit`

Generated experiments include:

- Delta Multiplier: `1` and `5`.
- Inputs: `200` and `20`.
- Patches: `"all"`, `"one"`, `"q1"`, `"q2"`, `"q3"`.

This creates configurations like:

- `mnist-1-200-all`
- `mnist-5-20-q3`

*Example: Winobias*

For the Winobias dataset:

- Input directory: `../data/wino_bias_txts`
- Model: `bert-base-uncased`
- Task: MLM
- Patches: `target-word`.

Generated experiments include:

- Delta Multiplier: `1` and `5`.
- Inputs: `200` and `20`.
- Vocabulary Tokens: `1`, `5`, and `10`.

This creates configurations like:

- `winobias-1-200-target-word-1`
- `winobias-5-20-target-word-10`

### Test Mode
If you want to test the pipeline with reduced configurations:
```bash
python experiments/prepare_experiments.py --default --test

```
This generates test versions of the experiments with fewer iterations and inputs for quick debugging.

## Run All the experiments in the `experiment_data` directory

To run all the test experiments prepared with the prepare_experiments.py script, use the following:

```bash
cd sh-scripts
bash test_experiments.sh
```

### <a name="exploration"></a>Experiments on input space exploration


##### ViT

To reproduce the experiments about **Input space exploration** paragraph in Section 4, including time needed to run 1000 iterations of SiMEC/SiMExp and color variance of the produced images, run:
```
bash vit_exploration.sh
```
This scripts also run the interpretation over the exploration results. Results can be found in `res/input-space-exploration`.

To run the perturbation-based baseline, run:
```
bash vit_baseline.sh
```

##### BERT
To reproduce the experiments about **Input space exploration** paragraph in Section 4, including time needed to run 1000 iterations of SiMEC/SiMExp, run:

```
bash bert_exploration.sh
```
This scripts also run the interpretation over the exploration results. Results can be found in `res/input-space-exploration`.

### Experiments on feature importance

##### BERT

To reproduce the experiments about **Feature importance-based explanations** from Section 4, run:
```
bash bert_feature_importance.sh
```

To reproduce and analyze the results of the baselines (Attention Rollout and the Relevancy method), run:
```
bash bert_baseline.sh
```

### Feature importance interpretability examples

To reproduce Figure 1 and Figure 2 in the **Feature Importance** paragraph in Section 3.2 (**Interpretability**), run:
```
bash feature_importance_interpretation_example.sh
```
Results can be found in `res/examples/feature-importance`.

### ViT exploration example

To reproduce Figure 3 in Section 3.2 (**Interpretation of input space exploration**), run:
```
 bash vit_interpretation_example.sh
```
Results can be found in `res/examples/input-space-exploration`. Specifically, the first image is just the original MNIST image in `mnist_imgs/example-/exploration`, and the other two images are the 1000th interpretation in simec and simexp experiments in `res/examples/input-space-exploration`.

### Plots in "Using interpretation outputs as alternative prompts"

To reproduce Figure 4 from Section 4.1 and Figures 1 and 2 in the Supplementary Materials, it is necessary to have exploration results for all experiments. If you do not have them, first follow the instructions at [Experiments on input space exploration](#exploration) section.

Once you have all exploration results, run:
```
bash eq_class_probas_analysis.sh
```
The resulting plots can be found in `res/plot_analysis/`.