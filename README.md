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

### Experiments on input space exploration


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
