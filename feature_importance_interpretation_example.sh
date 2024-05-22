cd experiments
echo "BERT Feature Importance Intepretation Examples"
python3 bert_feature_importance.py --objective cls --exp-name bert-cls --txt-dir ../hate_explain/example/feature-importance --model-name ctoraman/hate-speech-bert  --out-dir ../res/examples
python3 bert_feature_importance.py --objective mask --exp-name bert-msk --txt-dir ../mlm_data/example/feature-importance --model-name bert-base-uncased  --out-dir ../res/examples

echo "ViT Feature Importance Intepretation Example"
python3 vit_feature_importance.py --exp-name vit --img-dir ../mnist_imgs/example/feature-importance --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/examples
