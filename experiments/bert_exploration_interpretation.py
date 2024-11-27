"""
A module for exploring the input space of BERT models. It includes functionalities to visualize
the impact of individual tokens and their equivalence classes on the model's predictions, 
and to experiment with different configurations and equivalence classes.
"""

import argparse
import os
from tqdm import tqdm
from transformers import BertTokenizerFast, logging
import torch
from numpy import around
from experiments_utils import (
    load_bert_model,
    deactivate_dropout_layers,
    load_raw_sents,
    load_raw_sent,
    load_object,
    save_object,
    save_json,
    load_json,
)


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
    class_map: dict = None,
    txt_out_dir: str = ".",
    capping: str = "",
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

    def save_intepretation():
        json_result = {}
        json_result["sentence_id"] = sentence[1]
        json_result["sentence"] = sentence[0]
        json_result["tokenized_original_sentence"] = original_sentence
        json_result["target_token_id"] = keep_constant_id
        json_result["target_token"] = keep_constant_txt
        json_result["target_token_pred"] = (str_pred, str_pred_capped)
        json_result["eq_class_words"] = [
            (k, v)
            for k, v in (
                eq_class if len(eq_class) > 0 else enumerate(original_sentence)
            )
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
            file.write(str_res, encoding="utf-8")
        save_json(json_fname, json_result)
        stats_path = os.path.join(
            txt_out_dir,
            f"{iteration}-stats.json",
        )
        save_json(
            stats_path,
            json_stats,
        )

    str_pred_capped = "exploration-capping"  # just for return values purposes

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
            padding=True,
        )["input_ids"]
        .squeeze()
        .to(device)
    )
    original_sentence = tokenizer.convert_ids_to_tokens(original_sentence_ids)
    modified_sentence_ids = original_sentence_ids.clone().to(device)
    json_stats = {}

    model.eval()
    with torch.no_grad():
        if capping:
            min_cap = load_object(os.path.join(capping, "min_distribution.pkl")).to(
                device
            )
            max_cap = load_object(os.path.join(capping, "max_distribution.pkl")).to(
                device
            )
            # cap input embeddings to bring them back to what the decoder knows
            capped_input_embedding = input_embedding.clone().to(device)
            capped_input_embedding[capped_input_embedding < min_cap] = min_cap.repeat(
                (1, capped_input_embedding.size(1), 1)
            )[capped_input_embedding < min_cap]
            capped_input_embedding[capped_input_embedding > max_cap] = max_cap.repeat(
                (1, capped_input_embedding.size(1), 1)
            )[capped_input_embedding > max_cap]

        if mask_or_cls == "mask":
            # register prediction without cut back in allowed range
            mlm_pred = decoder(output_embedding)[0]
            json_stats["pre-cap-sentence"] = " ".join(
                tokenizer.convert_ids_to_tokens(torch.argmax(mlm_pred, dim=-1))
            )
            str_pred = tokenizer.convert_ids_to_tokens(
                [torch.argmax(mlm_pred[keep_constant_id]).item()]
            )[0]
            json_stats["pre-cap-probas"] = [
                (tokenizer.convert_ids_to_tokens([v])[0], p.item())
                for v, p in zip(
                    mlm_pred[keep_constant_id].topk(5).indices,
                    mlm_pred[keep_constant_id].topk(5).values,
                )
            ]
            if capping:
                # register prediction with cut back in allowed range
                mlm_pred_capped = decoder(
                    model.bert.encoder(capped_input_embedding)[0]
                )[0]
                json_stats["cap-sentence"] = " ".join(
                    tokenizer.convert_ids_to_tokens(
                        torch.argmax(mlm_pred_capped, dim=-1)
                    )
                )
                str_pred_capped = tokenizer.convert_ids_to_tokens(
                    [torch.argmax(mlm_pred_capped[keep_constant_id]).item()]
                )[0]
                json_stats["cap-probas"] = [
                    (tokenizer.convert_ids_to_tokens([v])[0], p.item())
                    for v, p in zip(
                        mlm_pred_capped[keep_constant_id].topk(5).indices,
                        mlm_pred_capped[keep_constant_id].topk(5).values,
                    )
                ]
            # register equivalence class words probabilities with and without cutting back
            for idx, w in (
                eq_class if len(eq_class) > 0 else enumerate(original_sentence)
            ):
                if w not in ["[CLS]", "[MASK]", "[SEP]"]:

                    if capping:
                        json_stats[f"cap-probas-{w}"] = [
                            (tokenizer.convert_ids_to_tokens([v])[0], p.item())
                            for v, p in zip(
                                mlm_pred_capped[idx].topk(5).indices,
                                mlm_pred_capped[idx].topk(5).values,
                            )
                        ]
                        modified_sentence_ids[idx] = torch.argmax(
                            mlm_pred_capped[idx]
                        ).item()
                    else:
                        json_stats[f"pre-cap-probas-{w}"] = [
                            (tokenizer.convert_ids_to_tokens([v])[0], p.item())
                            for v, p in zip(
                                mlm_pred[idx].topk(5).indices,
                                mlm_pred[idx].topk(5).values,
                            )
                        ]
                        modified_sentence_ids[idx] = torch.argmax(
                            mlm_pred_capped[idx]
                        ).item()
            # now we test whether the prediction has changed by processing the
            # sentence with alternative words taken from equivalence class.
            # This is done only on cut embeddings
            mlm_pred_modified = model(input_ids=modified_sentence_ids.unsqueeze(0))[
                0
            ].squeeze()
            str_preds_modified = tokenizer.convert_ids_to_tokens(
                [torch.argmax(mlm_pred_modified[keep_constant_id], dim=-1)]
            )[0]
            json_stats["mod-probas"] = [
                (tokenizer.convert_ids_to_tokens([v])[0], p.item())
                for v, p in zip(
                    mlm_pred_modified[keep_constant_id].topk(5).indices,
                    mlm_pred_modified[keep_constant_id].topk(5).values,
                )
            ]
            modified_sentence = tokenizer.convert_ids_to_tokens(modified_sentence_ids)
            json_stats["mod-sentence"] = modified_sentence
            json_stats["original-sentence"] = original_sentence

        else:
            # register prediction without cut back in allowed range
            mlm_pred = decoder(input_embedding)[0]
            json_stats["pre-cap-sentence"] = " ".join(
                tokenizer.convert_ids_to_tokens(torch.argmax(mlm_pred, dim=-1))
            )
            cls_pred = model.classifier(model.bert.pooler(output_embedding))
            str_pred = class_map[torch.argmax(cls_pred).item()]
            json_stats["pre-cap-probas"] = [
                (class_map[v], p.item())
                for v, p in enumerate(cls_pred[keep_constant_id])
            ]
            if capping:
                # register prediction with cut back in allowed range
                mlm_pred_capped = decoder(capped_input_embedding)[0]
                json_stats["cap-sentence"] = " ".join(
                    tokenizer.convert_ids_to_tokens(
                        torch.argmax(mlm_pred_capped, dim=-1)
                    )
                )
                cls_pred_capped = model.classifier(
                    model.bert.pooler(model.bert.encoder(capped_input_embedding)[0])
                )
                str_pred_capped = class_map[torch.argmax(cls_pred_capped).item()]
                json_stats["cap-probas"] = [
                    (class_map[v], p.item())
                    for v, p in enumerate(cls_pred_capped[keep_constant_id])
                ]
            # register equivalence class words probabilities with and without cutting back
            for idx, w in (
                eq_class if len(eq_class) > 0 else enumerate(original_sentence)
            ):
                if w not in ["[CLS]", "[MASK]", "[SEP]"]:
                    json_stats[f"pre-cap-probas-{w}"] = [
                        (tokenizer.convert_ids_to_tokens([v])[0], around(p.item(), 3))
                        for v, p in zip(
                            mlm_pred[idx].topk(5).indices,
                            mlm_pred[idx].topk(5).values,
                        )
                    ]
                    if capping:
                        json_stats[f"cap-probas-{w}"] = [
                            (
                                tokenizer.convert_ids_to_tokens([v])[0],
                                around(p.item(), 3),
                            )
                            for v, p in zip(
                                mlm_pred_capped[idx].topk(5).indices,
                                mlm_pred_capped[idx].topk(5).values,
                            )
                        ]
                        modified_sentence_ids[idx] = torch.argmax(
                            mlm_pred_capped[idx]
                        ).item()
                    else:
                        json_stats[f"pre-cap-probas-{w}"] = [
                            (
                                tokenizer.convert_ids_to_tokens([v])[0],
                                around(p.item(), 3),
                            )
                            for v, p in zip(
                                mlm_pred[idx].topk(5).indices,
                                mlm_pred[idx].topk(5).values,
                            )
                        ]
                        modified_sentence_ids[idx] = torch.argmax(mlm_pred[idx]).item()

            # now we test whether the prediction has changed by processing the
            # sentence with alternative words taken from equivalence class.
            # This is done only on cut embeddings
            cls_pred_modified = model(input_ids=modified_sentence_ids.unsqueeze(0))[0]
            str_preds_modified = class_map[
                torch.argmax(cls_pred_modified, dim=-1).item()
            ]
            json_stats["mod-probas"] = [
                (class_map[v], p.item())
                for v, p in enumerate(cls_pred_modified[keep_constant_id])
            ]
            modified_sentence = tokenizer.convert_ids_to_tokens(modified_sentence_ids)
            json_stats["mod-sentence"] = modified_sentence
            json_stats["original-sentence"] = original_sentence

    save_intepretation()

    if capping:
        return str_pred_capped, str_preds_modified
    return str_pred, str_preds_modified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, choices=["cls", "mask"], required=True)
    parser.add_argument("--pkl-dir", type=str, required=True)
    parser.add_argument("--txt-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--cap-ex", action="store_true")

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    txts, names = load_raw_sents(args.txt_dir)
    eq_class_words = load_json(os.path.join(args.txt_dir, "config.json"))
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

    obj = "msk" if args.objective == "mask" else "cls"

    for res_dir in os.listdir(args.pkl_dir):
        if res_dir.startswith("sime") and obj in res_dir:
            res_path = os.path.join(args.pkl_dir, res_dir)
            print(res_dir)

            print("\tPreprocessing text...")
            for idx, txt in enumerate(txts):
                tokenized_input = bert_tokenizer(
                    txt,
                    return_tensors="pt",
                    return_attention_mask=False,
                    add_special_tokens=False,
                    padding=True,
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
                            for ind, wrd in zip(
                                eq_class_word_ids, eq_class_words[names[idx]]
                            )
                        ],
                        key=lambda x: x[0],
                    ),
                }

            print("\tInterpretation phase")

            with torch.no_grad():
                for txt_dir in os.listdir(res_path):
                    if os.path.isdir(os.path.join(res_path, txt_dir)):
                        dirs = [
                            filename
                            for filename in os.listdir(os.path.join(res_path, txt_dir))
                            if os.path.isfile(os.path.join(res_path, txt_dir, filename))
                            and filename.lower().endswith(".pkl")
                        ]
                        predictions = {}
                        for filename in tqdm(dirs, desc=f"Reading {txt_dir}"):
                            res = load_object(os.path.join(res_path, txt_dir, filename))
                            pred, decoded_pred = interpret(
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
                                capping=res_path if args.cap_ex else "",
                                device=device,
                            )
                            predictions[res["iteration"]] = pred == decoded_pred
                        save_json(
                            os.path.join(res_path, txt_dir, "pred-stats.json"),
                            predictions,
                        )


if __name__ == "__main__":
    main()
