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
from transformers import BertTokenizerFast
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


def generate_combined_sentences(
    original_tokenized_sentence: list,
    eq_class_words: dict,
    tokenizer: BertTokenizerFast,
    max_num: int = None,
) -> tuple:
    """
    Generate new sentences by combining tokens from equivalent class words.

    Args:
        original_tokenized_sentence (list): The original tokenized sentence.
        eq_class_words (dict): A dictionary mapping indices to lists of equivalent tokens.
        tokenizer (BertTokenizerFast): The tokenizer used for BERT model.
        max_num (int, optional): Maximum number of sentence combinations to generate.

    Returns:
        tuple: Returns a tuple of two lists; the first contains the modified sentences as strings,
               and the second contains their token ids as a torch Tensor.
    """
    words_lists = [
        [k[1], *[el[1] for el in v if el[1] != k[1]]]
        for k, v in list(eq_class_words.items())
    ]
    combinations = list(itertools.product(*words_lists))
    if max_num:
        if max_num <= len(combinations):
            combinations_ids = random.choice(
                range(len(combinations)), max_num, replace=False
            )
            combinations = [
                c for i, c in enumerate(combinations) if i in combinations_ids
            ]

    mod_sentences, mod_sentences_idx = [], []
    list_idx = [idx for idx, _ in list(eq_class_words.keys())]
    print("Computing combinations ...")
    for combo in combinations:
        s = []
        i = 0
        for idx, w in enumerate(original_tokenized_sentence):
            if idx in list_idx:
                s.append(combo[i])
                i += 1
            else:
                s.append(w)
        mod_sentences.append(" ".join(s))
        mod_sentences_idx.append(tokenizer.convert_tokens_to_ids(s))
    return combinations, mod_sentences, torch.Tensor(mod_sentences_idx).int()


def interpret(
    sent_filename: tuple,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    tokenizer: BertTokenizerFast,
    input_embedding: torch.Tensor,
    output_embedding: torch.Tensor,
    eq_class_words_ids: dict,
    mask_or_cls: str,
    iteration: int,
    device: torch.device,
    min_cap: torch.Tensor,
    max_cap: torch.Tensor,
    class_map: dict = None,
    txt_out_dir: str = ".",
) -> None:
    """
    Interpret the results of sentence exploration and generate a report.

    This function analyzes the impact of individual tokens and their equivalence classes
    on the model's predictions. It saves the findings in both text and JSON formats.

    Args:
        sent_filename (tuple): Tuple containing the directory and file name of the sentence.
        model (torch.nn.Module): The trained BERT model.
        decoder (torch.nn.Module): A module to decode the embeddings into predictions.
        tokenizer (BertTokenizerFast): The tokenizer associated with the BERT model.
        input_embedding (torch.Tensor): The input embedding tensor.
        output_embedding (torch.Tensor): The output embedding tensor after passing through the model.
        eq_class_words_ids (dict): A dictionary mapping words to their equivalence classes and indices.
        mask_or_cls (str): The exploration objective, 'mask' for masked language modeling or 'cls' for classification.
        iteration (int): The current iteration number of the exploration.
        device (torch.device): The device (CPU/GPU) on which the computation is to be performed.
        top_k (int): Number of top predictions to consider for each token in the equivalence class.
        max_n_comb (int): Maximum number of token combinations to generate and analyze.
        class_map (dict, optional): Mapping of class indices to class names, used for classification tasks.
        txt_out_dir (str): Directory where the interpretative results will be saved.

    Returns:
        None. This function saves the exploration results to files and does not return any value.
    """
    txts_dir, sent_name = sent_filename
    eq_class, keep_constant = (
        eq_class_words_ids["eq_class_w"],
        eq_class_words_ids["keep_constant"],
    )
    keep_constant_id, keep_constant_txt = keep_constant
    sentence = load_raw_sent(txts_dir, sent_name)
    original_sentence_ids = (
        tokenizer(
            sentence[0],
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        .squeeze()
        .to(device)
    )
    print("Iteration: ", iteration)
    original_sentence = tokenizer.convert_ids_to_tokens(original_sentence_ids)
    allowed_tokens = get_allowed_tokens(tokenizer)

    model.eval()
    with torch.no_grad():
        # cap input embeddings to bring them back to what the decoder knows
        capped_input_embedding = input_embedding.clone().to(device)
        capped_input_embedding[capped_input_embedding < min_cap] = min_cap[
            capped_input_embedding < min_cap
        ]
        capped_input_embedding[capped_input_embedding > max_cap] = max_cap[
            capped_input_embedding > max_cap
        ]
        if mask_or_cls == "mask":
            mlm_pred = decoder(output_embedding)[0]
            mlm_pred[:, allowed_tokens] = mlm_pred[:, allowed_tokens] * 100
            print(
                "Pre-capped decoded sentence "
                + " ".join(
                    tokenizer.convert_ids_to_tokens(torch.argmax(mlm_pred, dim=-1))
                )
            )
            str_pred = tokenizer.convert_ids_to_tokens(
                [torch.argmax(mlm_pred[keep_constant_id]).item()]
            )[0]
            mlm_pred_capped = decoder(model.bert.encoder(capped_input_embedding)[0])[0]
            mlm_pred_capped[:, allowed_tokens] = (
                mlm_pred_capped[:, allowed_tokens] * 100
            )
            print(
                "Capped decoded sentence "
                + " ".join(
                    tokenizer.convert_ids_to_tokens(
                        torch.argmax(mlm_pred_capped, dim=-1)
                    )
                )
            )
            str_pred_capped = tokenizer.convert_ids_to_tokens(
                [torch.argmax(mlm_pred_capped[keep_constant_id]).item()]
            )[0]

        else:
            mlm_pred_capped = decoder(capped_input_embedding)[0]
            print(
                "Capped decoded sentence "
                + " ".join(
                    tokenizer.convert_ids_to_tokens(
                        torch.argmax(mlm_pred_capped, dim=-1)
                    )
                )
            )
            cls_pred = model.classifier(model.bert.pooler(output_embedding))
            str_pred = class_map[torch.argmax(cls_pred).item()]
            cls_pred_capped = model.classifier(
                model.bert.pooler(model.bert.encoder(capped_input_embedding)[0])
            )
            str_pred_capped = class_map[torch.argmax(cls_pred_capped).item()]

        modified_sentence_ids = original_sentence_ids.clone().to(device)
        for idx, w in eq_class if len(eq_class) > 0 else enumerate(original_sentence):
            if w not in ["[CLS]", "[MASK]", "[SEP]"]:
                print(
                    f"First top 5 probabilities after cap: {[(tokenizer.convert_ids_to_tokens([v])[0], around(p.item(),3)) for v,p in zip(mlm_pred_capped[idx].topk(5).indices, mlm_pred_capped[idx].topk(5).values)]}"
                )
                print("--")
                modified_sentence_ids[idx] = torch.argmax(mlm_pred_capped[idx]).item()

        if mask_or_cls == "mask":
            modified_sentence_ids[keep_constant_id] = torch.argmax(
                mlm_pred[keep_constant_id]
            ).item()

        if mask_or_cls == "mask":
            pred = model(input_ids=modified_sentence_ids.unsqueeze(0))
            mlm_pred = pred[0].squeeze()
            mlm_pred[:, allowed_tokens] = mlm_pred[:, allowed_tokens] * 100
            str_preds_modified = tokenizer.convert_ids_to_tokens(
                [torch.argmax(mlm_pred[keep_constant_id], dim=-1)]
            )[0]

            for idx, w in (
                eq_class if len(eq_class) > 0 else enumerate(original_sentence)
            ):
                if w not in ["[CLS]", "[MASK]", "[SEP]"]:
                    print(
                        f"First top 5 probabilities after cap: {[(tokenizer.convert_ids_to_tokens([v])[0], around(p.item(),3)) for v,p in zip(mlm_pred[idx].topk(5).indices, mlm_pred[idx].topk(5).values)]}"
                    )
                    print("--")
                    modified_sentence_ids[idx] = torch.argmax(mlm_pred[idx]).item()
        else:
            cls_pred = model(input_ids=modified_sentence_ids.unsqueeze(0))[0]
            str_preds_modified = [
                class_map[r.item()] for r in torch.argmax(cls_pred, dim=-1)
            ][0]

    modified_sentence = tokenizer.convert_ids_to_tokens(modified_sentence_ids)

    json_result = {}
    json_result["sentence_id"] = sentence[1]
    json_result["sentence"] = sentence[0]
    json_result["tokenized_original_sentence"] = original_sentence
    json_result["target_token_id"] = keep_constant_id
    json_result["target_token"] = keep_constant_txt
    json_result["target_token_pred"] = (str_pred, str_pred_capped)
    json_result["eq_class_words"] = [
        (k, v)
        for k, v in (eq_class if len(eq_class) > 0 else enumerate(original_sentence))
    ]
    json_result["modified_sentence"] = modified_sentence
    json_result["modified_sentence_pred"] = str_preds_modified

    str_res = (
        f"{json_result['sentence_id']}: {json_result['sentence']}\n"
        + f"Target token to keep constant: {json_result['target_token']}, predicted as '{json_result['target_token_pred'][0]}', and '{json_result['target_token_pred'][1]}' if capped\n"
        + f"Equivalence class exploration for the following words: {', ' .join(f'{el[1]} [{el[0]}]' for el in json_result['eq_class_words'])}\n"
        + "\n\n Modified sentence and respective prediction\n"
        + f"{' '.join(json_result['modified_sentence'])}\t{json_result['modified_sentence_pred']}"
    )

    if not os.path.exists(txt_out_dir):
        os.makedirs(txt_out_dir)
    fname = os.path.join(
        txt_out_dir,
        f"{iteration}-{str_pred}-{str_pred_capped}-{str_preds_modified}.txt",
    )
    json_fname = os.path.join(
        txt_out_dir,
        f"{iteration}-{str_pred}-{str_pred_capped}-{str_preds_modified}.json",
    )
    with open(fname, "w") as file:
        file.write(str_res)
    with open(json_fname, "w") as file:
        json.dump(json_result, file)


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

    bert_tokenizer, bert_model = load_bert_model(
        args.model_name, mask_or_cls=args.objective, device=device
    )
    deactivate_dropout_layers(bert_model)
    bert_model = bert_model.to(device)

    str_time = time.strftime("%Y%m%d-%H%M%S")
    # str_time = "20240514-065725"
    res_path = os.path.join(
        args.out_dir, "input-space-exploration", args.exp_name + "-" + str_time
    )

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(os.path.join(res_path, "params.json"), "w") as file:
        json.dump(vars(args), file)

    print("\tMeasuring input distribution...")
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

    embeddings = torch.concat([s.permute(0, 2, 1) for s in sentence_embeddings], dim=-1)
    min_embeddings = torch.min(embeddings, dim=-1).values
    max_embeddings = torch.max(embeddings, dim=-1).values

    save_object(
        obj=min_embeddings.cpu(),
        filename=os.path.join(res_path, "min_distribution.pkl"),
    )
    save_object(
        obj=max_embeddings.cpu(),
        filename=os.path.join(res_path, "max_distribution.pkl"),
    )

    for idx, txt in enumerate(txts):

        print(f"Sentence:{names[idx]}\t{idx+1}/{len(txts)}")

        print("\tExploration phase")

        explore(
            same_equivalence_class=args.exp_type == "same",
            input_embedding=sentence_embeddings[idx],
            model=bert_model.bert.encoder,
            eq_class_emb_ids=(
                eq_class_word_ids if len(eq_class_word_ids) > 0 else None
            ),
            pred_id=keep_constant,
            device=device,
            threshold=args.threshold,
            n_iterations=args.iter,
            out_dir=os.path.join(res_path, names[idx]),
            save_each=args.save_each,
        )

    print("\tInterpretation phase")

    with torch.no_grad():
        for txt_dir in os.listdir(res_path):
            if os.path.isdir(os.path.join(res_path, txt_dir)):
                for filename in os.listdir(os.path.join(res_path, txt_dir)):
                    if os.path.isfile(
                        os.path.join(res_path, txt_dir, filename)
                    ) and filename.lower().endswith(".pkl"):
                        res = load_object(os.path.join(res_path, txt_dir, filename))
                        min_embeddings = load_object(
                            os.path.join(res_path, "min_distribution.pkl")
                        )
                        max_embeddings = load_object(
                            os.path.join(res_path, "max_distribution.pkl")
                        )
                        interpret(
                            sent_filename=(args.txt_dir, txt_dir),
                            model=bert_model.to(device),
                            decoder=(
                                bert_model.cls
                                if args.objective == "mask"
                                else bert_model.decoder
                            ),
                            tokenizer=bert_tokenizer,
                            class_map=class_map,
                            input_embedding=res["input_embedding"].to(device),
                            output_embedding=res["output_embedding"].to(device),
                            mask_or_cls=args.objective,
                            iteration=res["iteration"],
                            eq_class_words_ids=eq_class_words_and_ids[txt_dir],
                            txt_out_dir=os.path.join(
                                res_path, txt_dir, "interpretation"
                            ),
                            min_cap=min_embeddings.to(device),
                            max_cap=max_embeddings.to(device),
                            device=device,
                        )


if __name__ == "__main__":
    main()
