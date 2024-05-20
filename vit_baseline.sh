cd analysis
for folder in ../../mnist_imgs/*:
do
    python3 vit_perturbation.sh --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --img-dir ../mnist_imgs/$folder --timer 339 --pert-step 0.01 --out-dir ../res/mnist_perturbation
done
python3 mnist_perturbation_analysis.py --dir ../res/mnist_perturbation