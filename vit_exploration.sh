cd experiments
for folder in ../mnist_imgs/*
do
    python3 vit_exploration.py --exp-type same --exp-name simec-vit --img-dir ../mnist_imgs/$folder --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/mnist_experiment --iter 1000 --save-each 10
    python3 vit_exploration.py --exp-type diff --exp-name simexp-vit --img-dir ../mnist_imgs/$folder --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/mnist_experiment --iter 1000 --save-each 10
done
cd ../analysis
python3 time_analysis.py --dir ../res/mnist_experiment --iter 1000
python3 mnist_perturbation_analysis.py --dir ../res/mnist_experiment