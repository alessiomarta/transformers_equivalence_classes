from datasets import load_dataset
import torch
import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import gc
from transformers import BertTokenizerFast
import sys
sys.path.append("./analysis/Transformer_Explainability/BERT_rationale_benchmark/models")
sys.path.append("./analysis/Transformer_Explainability/BERT_explainability/modules/BERT")
sys.path.append("./analysis/Transformer_Explainability/")
from ExplanationGenerator import Generator
from BertForSequenceClassification import BertForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--sample", type=int, default = 200)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


if __name__ == "__main__":

    args = parse_args()
    device = args.device
    model_name = args.model_path
    test_sample_size = args.sample
    base_dir = f"./{args.dataset.lower()}_txts"
    x_label = "text"
    y_label = "hatespeech"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split = "train", trust_remote_code = True)
    testset = dataset.select_columns([x_label, y_label])

    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name)

    explanations = Generator(bert_model)

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        _, indices = train_test_split(list(range(len(testset))), test_size = test_sample_size, random_state=42, stratify=testset[y_label])
        subset = torch.utils.data.Subset(testset, indices)
        data_test = subset.__getitems__(list(range(test_sample_size)))
        X_test = list(map(lambda tup: tup[x_label], data_test))
        y_test = list(map(lambda tup: tup[y_label], data_test))
    else:
        indices = list(range(len(testset)))
        X_test = testset[x_label]
        y_test = testset[y_label]

    y_test = list(map(lambda x: int(x) if isinstance(x, float) else x, y_test))
    classes = list(set(y_test))

    for y,label in enumerate(classes):

        # Set saving directory
        label_dir = os.path.join(base_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Set config file
        configs = defaultdict(dict)
        attributions = defaultdict(dict)

        # Filter by label
        y_indices = [i for i,c in zip(indices, y_test) if c == label]
        X = [s for s,c in zip(X_test, y_test) if c == label]

        for i,sent in enumerate(X):
            fname = f"sentence_{y_indices[i]}"
            print(fname)

            # Save text
            with open(os.path.join(label_dir, fname + ".txt"), "w", encoding = "utf-8") as f:
                f.write(sent)

            # Compute attribution
            encoded_input = bert_tokenizer(sent, return_tensors = "pt", add_special_tokens = False)
            pred = bert_model(**encoded_input)
            pred = pred.logits.detach().numpy().flatten().argmax()

            expl = explanations.generate_LRP(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask, start_layer=0)[0]
            expl = (expl - expl.min()) / (expl.max() - expl.min())
            expl = expl.detach().cpu().numpy()

            sentence_tokens = bert_tokenizer.tokenize(sent)
            tot_words = len(sentence_tokens)
            N_words = [1, tot_words // 4, tot_words // 2, 3*(tot_words//4), tot_words]
            perc = ["one", "q1", "q2", "q3", "all"]

            sorted_idx = np.argsort(expl)[::-1]
            sorted_scores = np.sort(expl)[::-1]

            for n,a in zip(N_words, perc):
                if n == tot_words:
                    configs[a][fname] = []
                    attributions[a][fname] = {
                        "idx": [],
                        "att": [],
                        "tok": sentence_tokens
                    }
                else:
                    configs[a][fname] = sorted_idx[:n].tolist()
                    attributions[a][fname] = {
                        "idx": sorted_idx[:n].tolist(),
                        "att": sorted_scores[:n].tolist(),
                        "tok": [sentence_tokens[j] for j in sorted_idx[:n].tolist()]
                    }

            del expl
            gc.collect()

        for n,config in configs.items():
            with open(os.path.join(label_dir, f"config_{n}.json"), "w") as f:
                json.dump(config, f)

        for n,config in attributions.items():
            with open(os.path.join(label_dir, f"attrib_{n}.json"), "w") as f:
                json.dump(config, f)

    