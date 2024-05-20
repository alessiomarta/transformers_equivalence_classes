cd analysis
python3 bert_att_rollout.py --model ctoraman/hate-speech-bert --txt-dir ../hatexplain_data/sentences --out-dir ../res/bert-grad-rollout
python3 bert_evaluation.py --datapath ../hatexplain_data/sentences/token_importance.json --predpath ../res/bert-grad-rollout --txt-dir ../hatexplain_data/sentences --model-name ctoraman/hate-speech-bert
python3 bert_attention_exp.py --model ctoraman/hate-speech-bert --txt-dir ../hatexplain_data/sentences --out-dir ../res/bert-relevancy
python3 bert_evaluation.py --datapath ../hatexplain_data/sentences/token_importance.json --predpath ../res/bert-relevancy --txt-dir ../hatexplain_data/sentences --model-name ctoraman/hate-speech-bert