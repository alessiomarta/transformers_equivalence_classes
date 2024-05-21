cd experiments
echo "Feature Importance Intepretation Example"
python3 vit_feature_importance.py --exp-name vit --img-dir ../mnist_imgs/example/feature-importance --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/examples

echo "Exploration Interpretation Example"
echo "Exploration phase..."
python3 vit_exploration.py --exp-type same --exp-name simec-vit --img-dir ../mnist_imgs/example/exploration --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/examples --iter 1000 --save-each 10
python3 vit_exploration.py --exp-type diff --exp-name simexp-vit --img-dir ../mnist_imgs/example/exploration --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/examples --iter 1000 --save-each 10

echo "Interpretation phase..."
python vit_exploration_interpretation.py --model-path models/vit/model_20.pt --config-path models/vit/config.json --img-dir data/MNIST/MNIST/images --pkl-dir ../res/examples/input-space-exploration 