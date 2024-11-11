cd experiments
echo "BERT CLS Exploration ..."
python3 bert_exploration.py --objective cls --exp-type same --exp-name simec-bert-cls --model-name ctoraman/hate-speech-bert --out-dir ../res/ --txt-dir ../hate_explain/generated --iter 1000 --save-each 10
python3 bert_exploration.py --objective cls --exp-type diff --exp-name simexp-bert-cls --model-name ctoraman/hate-speech-bert --out-dir ../res/ --txt-dir ../hate_explain/generated --iter 1000 --save-each 10

echo "BERT MSK Exploration ..."
python3 bert_exploration.py --objective mask --exp-type same --exp-name simec-bert-msk --model-name bert-base-uncased --out-dir ../res/ --txt-dir ../mlm_data/generated --iter 1000 --save-each 10
python3 bert_exploration.py --objective mask --exp-type diff --exp-name simexp-bert-msk --model-name bert-base-uncased --out-dir ../res/ --txt-dir ../mlm_data/generated --iter 1000 --save-each 10

echo "BERT MSK Interpretation ..."
python3 bert_exploration_interpretation.py  --objective mask --pkl-dir ../res/input-space-exploration --txt-dir ../mlm_data/generated --model-name bert-base-uncased

echo "BERT CLS Interpretation ..."
python3 bert_exploration_interpretation.py  --objective cls --pkl-dir ../res/input-space-exploration --txt-dir ../hate_explain/generated --model-name ctoraman/hate-speech-bert

cd ../analysis
python3 time_analysis.py --dir ../res/input-space-exploration --iter 1000 --exp cls
python3 time_analysis.py --dir ../res/input-space-exploration --iter 1000 --exp msk