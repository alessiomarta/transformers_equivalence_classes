cd experiments
python3 bert_feature_importance.py --objective cls --exp-name bert-cls --txt-dir ../hate_explain/test/sentences --model-name ctoraman/hate-speech-bert --out-dir ../res
cd ../analysis
python3 results_reset_dir.py --old ../res/bert-cls --new ../res/bert_ours
python3 bert_evaluation.py --datapath ../hate_explain/test/sentences/token_importance.json --predpath ../res/bert_ours --txt-dir ../hate_explain/test/sentences --model-name ctoraman/hate-speech-bert