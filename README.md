# SIMEC and SIMEXP experiments

Until the `simec` package is not published on PyPI, install it in editable mode:
```
pip install -e simec
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
