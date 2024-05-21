"""
A module for exploring the input space of BERT models. It includes functionalities to visualize
the impact of individual tokens and their equivalence classes on the model's predictions, 
and to experiment with different configurations and equivalence classes.
"""

import argparse
import os
import time
import json
import itertools
from numpy import random
from transformers import BertTokenizerFast, logging
import torch
from numpy import around
from simec.logics import explore
from experiments_utils import (
    load_bert_model,
    get_allowed_tokens,
    deactivate_dropout_layers,
    load_raw_sents,
    load_raw_sent,
    load_object,
    save_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-type", type=str, choices=["same", "diff"], required=True)
    parser.add_argument("--objective", type=str, choices=["cls", "mask"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--save-each", type=int, default=1)
    parser.add_argument("--txt-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    txts, names = load_raw_sents(args.txt_dir)
    eq_class_words = json.load(open(os.path.join(args.txt_dir, "config.json"), "r"))
    eq_class_words_and_ids = eq_class_words.copy()
    class_map = None
    if args.objective == "cls":
        class_map = {int(k): v for k, v in eq_class_words["class-map"].items()}
    logging.set_verbosity_error()
    bert_tokenizer, bert_model = load_bert_model(
        args.model_name, mask_or_cls=args.objective, device=device
    )
    deactivate_dropout_layers(bert_model)
    bert_model = bert_model.to(device)

    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, "input-space-exploration", args.exp_name + "-" + str_time
    )

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(os.path.join(res_path, "params.json"), "w") as file:
        json.dump(vars(args), file)

    sentence_embeddings = []
    for idx, txt in enumerate(txts):
        tokenized_input = bert_tokenizer(
            txt,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        ).to(device)
        # finding token of which to keep the prediction constant
        keep_constant = 0
        if args.objective == "mask":
            keep_constant = [
                i
                for i, el in enumerate(tokenized_input["input_ids"].squeeze())
                if el == bert_tokenizer.mask_token_id
            ][0]

        # finding token of which to explore the equivalence class
        # words could be not ordered in the list, thus they need to be aligned
        eq_class_word_ids = []
        for el1 in eq_class_words[names[idx]]:
            for i, el2 in zip(
                tokenized_input.word_ids(),
                bert_tokenizer.convert_ids_to_tokens(
                    tokenized_input["input_ids"].squeeze()
                ),
            ):
                # take into account multiple repetitions of the same word
                if el1 == el2 and i not in eq_class_word_ids:
                    eq_class_word_ids.append(i)

        # object to make interpretation easier
        eq_class_words_and_ids[names[idx]] = {
            "keep_constant": (
                keep_constant,
                "[CLS]" if keep_constant == 0 else "[MASK]",
            ),
            "eq_class_w": sorted(  # this needs to be in the same order as it appears in the original sentence
                [
                    (ind, wrd)
                    for ind, wrd in zip(eq_class_word_ids, eq_class_words[names[idx]])
                ],
                key=lambda x: x[0],
            ),
        }
        sentence_embeddings.append(
            bert_model.bert.embeddings(**tokenized_input).to(device)
        )

    for idx, txt in enumerate(txts):

        print(f"Sentence:{names[idx]}\t{idx+1}/{len(txts)}")

        print("\tExploration phase")

        explore(
            same_equivalence_class=args.exp_type == "same",
            input_embedding=sentence_embeddings[idx],
            model=bert_model.bert.encoder,
            eq_class_emb_ids=(
                None
                if eq_class_words_and_ids[names[idx]]["eq_class_w"] == []
                else [t[0] for t in eq_class_words_and_ids[names[idx]]["eq_class_w"]]
            ),
            pred_id=eq_class_words_and_ids[names[idx]]["keep_constant"][0],
            device=device,
            threshold=args.threshold,
            n_iterations=args.iter,
            out_dir=os.path.join(res_path, names[idx]),
            save_each=args.save_each,
        )


if __name__ == "__main__":
    main()
