#!/bin/bash

python3 image_convert.py --dataset mnist --model-path ../models/mnist_experiment/model_final.pt --config-path ../models/mnist_experiment/config.json --sample 50
python3 image_convert.py --dataset cifar10 --model-path ../models/cifar10_experiment/model_final.pt --config-path ../models/cifar10_experiment/config.json --sample 50
python3 text_convert.py --dataset ucberkeley-dlab/measuring-hate-speech --xlabel text --ylabel hatespeech --model-path ctoraman/hate-speech-bert --sample 50
python3 text_convert.py --dataset uclanlp/wino_bias --subset type1_pro --xlabel tokens --mask coreference_clusters --explore coreference_clusters --model-path gaunernst/bert-small-uncased --sample 50