cd experiments
echo "Feature Importance Intepretation Examples"
python3 bert_feature_importance.py --exp-name bert-cls --img-dir ../hate_explain/example/feature-importance --model-name ctoraman/hate-speech-bert  --out-dir ../res/examples
python3 bert_feature_importance.py --exp-name bert-msk --img-dir ../mlm_data/example/feature-importance --model-name bert-base-uncased  --out-dir ../res/examples