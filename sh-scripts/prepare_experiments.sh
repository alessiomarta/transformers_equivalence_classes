cd ..
python experiments/prepare_experiment.py \
    --exp_name "test-experiment" \
    --iterations 100 \
    --inputs 10 \
    --patches "one" \
    --orig_data_dir ../data/mnist_imgs \
    --model_path ../models/vit \
    --exp_dir ../experiments

