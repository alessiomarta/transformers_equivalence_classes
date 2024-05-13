import json
import os
import argparse
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer
from experiments_utils import *
import numpy as np
from time import sleep


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type = str, required = True)
    parser.add_argument("--predpath", type = str, required = True)
    parser.add_argument("--txt-dir", type = str, required = True)
    parser.add_argument("--model-name", type = str, required = True)
    args = parser.parse_args()

    # python bert_evaluation.py --datapath ../../hatexplain_data/sentences/token_importance.json --predpath ../../results/bert-grad-rollout --txt-dir ../../hatexplain_data/sentences --model-name ctoraman/hate-speech-bert

    return args


def token_score_align(words, encoding, token_scores):

    word_scores = [0]*len(words)

    for j in range(len(encoding)):

        word_id = encoding.token_to_word(j+1)
        if word_id and j < len(token_scores):
            word_scores[word_id] += token_scores[j]

    return word_scores


def main():

    args = parse_args()
    datapath = args.datapath
    predpath = args.predpath
    txt_dir = args.txt_dir
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    count = 0

    with open(os.path.join(datapath), "r") as f:
        data = json.load(f)

    print("Number of annotations:", len(data))

    preds = {}
    for file in os.listdir(predpath):
        if file.endswith(".json"):
            with open(os.path.join(predpath, file), "r") as f:
                d = json.load(f)
            preds[file.replace(".json", "")] = d['tokens_imp']

    txts, names = load_raw_sents(txt_dir)

    print("Number of predictions:", len(preds))

    tokenized_texts = list(map(lambda s: s.split(), txts))

    encoded_txts = tokenizer(tokenized_texts, is_split_into_words = True, add_special_tokens = False, padding = True, return_tensors = "pt")

    similarities = []
    for name, txt, encoding in zip(names, tokenized_texts, encoded_txts.encodings):
        print("Document:", name)

        ground_truth = data[name]
        if name not in preds:
            print(f"{name} not in predictions")
            continue

        count += 1
        pred_scores = list(map(lambda tup: tup[0], preds[name]))

        if np.isnan(pred_scores[0]):
            continue

        word_scores = token_score_align(txt, encoding, pred_scores)

        try:
            assert len(ground_truth) == len(word_scores)
        except:
            raise Exception(f"Length of pred scores is {len(word_scores)}, while length of ground truth is {len(ground_truth)}")

        y_pred = np.array(word_scores) / max(word_scores)
        y_true = np.array(ground_truth) / max(ground_truth)

        L = 1 - cosine(y_pred, y_true)

        similarities.append(L)

    print("Average similarity:", np.mean(similarities))
    print("Final count:", count)


if __name__ == "__main__":
    main()
