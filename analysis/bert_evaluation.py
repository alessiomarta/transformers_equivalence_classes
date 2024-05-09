import json
import os
import argparse
from torch.nn import BCELoss, Softmax
from transformers import AutoTokenizer
from experiments_utils import *
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type = str, required = True)
    parser.add_argument("--predpath", type = str, required = True)
    parser.add_argument("--txt-dir", type = str, required = True)
    parser.add_argument("--model-name", type = str, required = True)
    args = parser.parse_args()

    # python bert_evaluation.py --datapath ../../hatexplain_data/sentences/token_importance.json --predpath ../../results/bert-grad-rollout --txt-dir ../../hatexplain_data/sentences --model-name ctoraman/hate-speech-bert

    return args


def token_score_align(sentence, encoding, word_scores):

    words = sentence.split()
    token_scores = [0]*len(encoding)

    for j in range(len(words)):

        token_ids = encoding.word_to_tokens(j)
        if token_ids:
            for k in token_ids:
                token_scores[k-1] = word_scores[j]

    return token_scores


def main():

    args = parse_args()
    datapath = args.datapath
    predpath = args.predpath
    txt_dir = args.txt_dir
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(os.path.join(datapath), "r") as f:
        data = json.load(f)

    preds = {}
    for file in os.listdir(predpath):
        if file.endswith(".json"):
            with open(os.path.join(predpath, file), "r") as f:
                d = json.load(f)
            preds[file.replace(".json", "")] = d['tokens_imp']

    txts, names = load_raw_sents(txt_dir)

    encoded_txts = tokenizer(txts, add_special_tokens = False)
    loss = BCELoss()
    softmax = Softmax()

    cross_entropies = []
    for name, txt, encoding in zip(names, txts, encoded_txts.encodings):
        print("Document:", name)

        word_scores = data[name]
        pred_scores = list(map(lambda tup: tup[0], preds[name]))
        token_scores = token_score_align(txt, encoding, word_scores)

        try:
            assert len(pred_scores) == len(token_scores)
        except:
            raise Exception(f"Length of pred scores is {len(pred_scores)}, while length of token scores is {len(token_scores)}")

        y_pred = softmax(torch.tensor(pred_scores))
        y_true = softmax(torch.tensor(token_scores))

        cross_entropies.append(loss(y_pred, y_true).item())

    print("Final BCE:", np.mean(cross_entropies))


if __name__ == "__main__":
    main()
