from datasets import load_dataset
import torch
import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import re
import argparse
import gc
from transformers import BertTokenizerFast
import sys

sys.path.append(
    "../analysis/Transformer_Explainability/BERT_rationale_benchmark/models"
)
sys.path.append(
    "../analysis/Transformer_Explainability/BERT_explainability/modules/BERT"
)
sys.path.append("../analysis/Transformer_Explainability/")
from ExplanationGenerator import Generator
from BertForSequenceClassification import BertForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True
    )  # "ucberkeley-dlab/measuring-hate-speech"
    parser.add_argument("--subset", type=str)
    parser.add_argument("--xlabel", type=str, required=True)  # "text"
    parser.add_argument("--ylabel", type=str)  # "hatespeech"
    parser.add_argument("--mask", type=str)
    parser.add_argument("--explore", type=str)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--sample", type=int, default=200)

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type

    if args.ylabel is None:
        if args.mask is None or args.explore is None:
            raise ValueError("Either ylabel or mask and explore must be specified")
    else:
        if args.mask or args.explore:
            print(
                "Both ylabel and mask/explore were passed. Priority is given to ylabel (text classification)"
            )

    return args


if __name__ == "__main__":

    args = parse_args()
    device = args.device
    model_name = args.model_path
    dataset_name = args.dataset.split("/")[-1]
    test_sample_size = args.sample
    base_dir = f"./{dataset_name}_txts"
    x_label = args.xlabel
    y_label = args.ylabel
    mask = args.mask
    explore = args.explore

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    try:
        dataset = load_dataset(
            args.dataset, name=args.subset, split="test", trust_remote_code=True
        )
    except ValueError:
        dataset = load_dataset(
            args.dataset, name=args.subset, split="train", trust_remote_code=True
        )

    if y_label is not None:
        testset = dataset.select_columns([x_label, y_label])
    else:
        testset = dataset.select_columns([x_label, mask, explore])

    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name)

    explanations = Generator(bert_model)

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        stratify = testset[y_label] if y_label else None
        _, indices = train_test_split(
            list(range(len(testset))),
            test_size=test_sample_size,
            random_state=42,
            stratify=stratify,
        )
        subset = torch.utils.data.Subset(testset, indices)
        data_test = subset.__getitems__(list(range(test_sample_size)))
        X_test = list(map(lambda tup: tup[x_label], data_test))
        y_test = list(map(lambda tup: tup[y_label], data_test)) if y_label else []
        masked = list(map(lambda tup: tup[mask], data_test)) if mask else []
        explored = list(map(lambda tup: tup[explore], data_test)) if explore else []
    else:
        indices = list(range(len(testset)))
        X_test = testset[x_label].to_list()
        y_test = testset[y_label].to_list() if y_label else []
        masked = testset[mask].to_list() if mask else []
        explored = testset[explore].to_list() if explore else []

    y_test = list(map(lambda x: int(x) if isinstance(x, float) else x, y_test))
    classes = list(set(y_test))

    if classes:

        for y, label in enumerate(classes):

            # Set saving directory
            label_dir = os.path.join(base_dir, str(label))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            # Set config file
            configs = defaultdict(dict)

            # Filter by label
            y_indices = [i for i, c in zip(indices, y_test) if c == label]
            X = [s for s, c in zip(X_test, y_test) if c == label]

            for i, sent in enumerate(X):
                fname = f"sentence_{y_indices[i]}"
                print(fname)

                # Save text
                with open(
                    os.path.join(label_dir, fname + ".txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(sent)

                # Compute attribution
                encoded_input = bert_tokenizer(
                    sent, return_tensors="pt", padding=True, add_special_tokens=True
                )
                pred = bert_model(**encoded_input)
                pred = pred.logits.detach().numpy().flatten().argmax()

                expl = explanations.generate_LRP(
                    input_ids=encoded_input.input_ids,
                    attention_mask=encoded_input.attention_mask,
                    start_layer=0,
                )[0]
                expl = (expl - expl.min()) / (expl.max() - expl.min())
                expl = expl.detach().cpu().numpy()

                sentence_tokens = encoded_input.encodings[0].tokens
                tot_words = len(sentence_tokens) - 2
                N_words = [
                    1,
                    tot_words // 4,
                    tot_words // 2,
                    3 * (tot_words // 4),
                    tot_words,
                ]
                perc = ["one", "q1", "q2", "q3", "all"]

                sorted_idx = np.argsort(expl[:-1])[::-1]
                sorted_scores = np.sort(expl[:-1])[::-1]
                sorted_scores = sorted_scores[sorted_idx != 0]
                sorted_idx = sorted_idx[sorted_idx != 0]

                for n, a in zip(N_words, perc):
                    if n == tot_words:
                        configs[a][fname] = {
                            "objective": 0,
                            "explore": [],
                            "attrib": [],
                            "tokens": sentence_tokens,
                        }
                    else:
                        configs[a][fname] = {
                            "objective": 0,
                            "explore": sorted_idx[:n].tolist(),
                            "attrib": sorted_scores[:n].tolist(),
                            "tokens": [
                                sentence_tokens[j] for j in sorted_idx[:n].tolist()
                            ],
                        }

                del expl
                gc.collect()

            for n, config in configs.items():
                with open(os.path.join(label_dir, f"config_{n}.json"), "w") as f:
                    json.dump(config, f)

    else:

        config = {}

        if mask == explore:
            masked = list(map(lambda tup: int(tup[-1]), masked))
            explored = list(map(lambda tup: int(tup[1]), explored))

        is_split_into_words = isinstance(X_test[0], list)

        X_tokenized = bert_tokenizer(
            X_test,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            is_split_into_words=is_split_into_words,
        )

        for i, (encoding, m_word, x_word) in enumerate(
            zip(X_tokenized.encodings, masked, explored)
        ):

            fname = f"sentence_{indices[i]}"
            print(fname)

            sent = " ".join(encoding.tokens)
            sent = re.sub(r" #+", "", sent)
            sent = re.sub(r"( \[PAD\])", "", sent).strip()

            with open(
                os.path.join(base_dir, fname + ".txt"), "w", encoding="utf-8"
            ) as f:
                f.write(sent)

            m_token = encoding.word_to_tokens(m_word)[0]
            x_tokens = encoding.word_to_tokens(x_word)
            x_tokens = list(range(x_tokens[0], x_tokens[1]))

            config[fname] = {
                "objective": m_token,
                "explore": x_tokens,
                "tokens": [encoding.tokens[j] for j in x_tokens],
                "masked": encoding.tokens[m_token],
            }

        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump(config, f)
