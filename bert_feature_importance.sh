cd experiments
python3 bert_feature_importance.py --objective cls --exp-name hatexplain_experiment --txt-dir ../hatexplain_data/sentences --model-name ctoraman/hate-speech-bert --out-dir ../res
cd ../analysis
python3 results_reset_dir.py --old ../res/hatexplain_experiment --new ../res/bert_ours
python3 bert_evaluation.py --datapath ../hatexplain_data/sentences/token_importance.json --predpath ../res/bert_ours --txt-dir ../hatexplain_data/sentences --model-name ctoraman/hate-speech-bert