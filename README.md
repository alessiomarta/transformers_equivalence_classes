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

### <a name="exploration"></a>Experiments on input space exploration


##### ViT

To reproduce the experiments about **Input space exploration** paragraph in Section 4, including time needed to run 1000 iterations of SiMEC/SiMExp and color variance of the produced images, run:
```
bash vit_exploration.sh
```

To run the perturbation-based baseline, run:
```
bash vit_baseline.sh
```

##### BERT
To reproduce the experiments about **Input space exploration**

```
bash bert_exploration.sh
```


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
Results can be found in `res/examples/feature-importance`

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