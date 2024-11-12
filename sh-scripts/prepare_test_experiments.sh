cd ..
echo "Preparing MNIST experiment"
python experiments/prepare_experiment.py \
    --exp_name mnist \
    --iterations 10 \
    --inputs 10 \
    --patches one \
    --orig_data_dir ../data/mnist_imgs \
    --model_path ../models/vit \
    --exp_dir ../experiments/test

echo "Preparing CIFAR experiment"
python experiments/prepare_experiment.py \
    --exp_name cifar \
    --iterations 10 \
    --inputs 10 \
    --patches one \
    --orig_data_dir ../data/cifar10_imgs \
    --model_path ../models/cifarTrain \
    --exp_dir ../experiments/test

echo "Preparing WINOBIAS experiment"
python experiments/prepare_experiment.py \
    --exp_name winobias \
    --iterations 10 \
    --inputs 10 \
    --patches target-word \
    --orig_data_dir ../data/wino_bias_txts \
    --model_path bert-base-uncased \
    --objective mlm \
    --exp_dir ../experiments/test

echo "Preparing HATESPEECH experiment"
python experiments/prepare_experiment.py \
    --exp_name hatespeech \
    --iterations 10 \
    --inputs 10 \
    --patches one \
    --orig_data_dir ../data/measuring-hate-speech_txts \
    --model_path ctoraman/hate-speech-bert \
    --exp_dir ../experiments/test

