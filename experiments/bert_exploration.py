"""
A module for exploring the input space of BERT models. It includes functionalities to visualize
the impact of individual tokens and their equivalence classes on the model's predictions, 
and to experiment with different configurations and equivalence classes.
"""

import argparse
import os
import time
import json
from transformers import BertTokenizerFast
import torch
from simec.logics import explore
from experiments_utils import (
    load_bert_model,
    get_allowed_tokens,
    deactivate_dropout_layers,
    load_raw_sents,
    load_raw_sent,
    load_object,
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
    class_map: dict = None,
    txt_out_dir: str = ".",
) -> None:
    """
    Interpret and document the exploration findings for a particular sentence and model configuration.

    Args:
        sent_filename: Tuple containing the directory and file name of the sentence.
        model: The trained BERT model.
        decoder: A module to decode the embeddings into predictions.
        tokenizer: The tokenizer associated with the BERT model.
        input_embedding: The input embedding tensor.
        output_embedding: The output embedding tensor after passing through the model.
        eq_class_words_ids: A dictionary with equivalence class word indices and the index to keep constant.
        mask_or_cls: The exploration objective, either 'mask' for masked language modeling or 'cls' for classification.
        iteration: The current iteration number of the exploration.
        class_map: Optional mapping of class indices to class names for classification tasks.
        txt_out_dir: Directory to save the interpretative results.

    Returns:
        None. Saves the results of the exploration in a text file.
    """
    txts_dir, sent_name = sent_filename
    eq_class, keep_constant = (
        eq_class_words_ids["eq_class_w"],
        eq_class_words_ids["keep_constant"],
    )
    keep_constant_id, keep_constant_txt = keep_constant
    sentence = load_raw_sent(txts_dir, sent_name)
    allowed_tokens = get_allowed_tokens(tokenizer)
    if mask_or_cls == "mask":
        mlm_pred = decoder(output_embedding)[0]
        mlm_pred[:, allowed_tokens] = mlm_pred[:, allowed_tokens] * 100
        str_pred = tokenizer.convert_ids_to_tokens(
            [torch.argmax(mlm_pred[keep_constant_id]).item()]
        )[0]
    else:
        mlm_pred = decoder(input_embedding)[0]
        cls_pred = model.classifier(model.bert.pooler(output_embedding))
        str_pred = class_map[torch.argmax(cls_pred).item()]

    str_res = f"{sentence[1]}: {sentence[0]}\n"
    str_res += f"Target token to keep constant: {keep_constant_txt}, predicted as '{str_pred}'\n"
    str_res += (
        "Equivalence class exploration for the following words: "
        + ", ".join([w for i, w in eq_class])
        + "\n"
    )
    for idx, w in eq_class:
        str_res += f"Equivalence class for '{w}' (first 5 words): "
        similar_words = tokenizer.convert_ids_to_tokens(mlm_pred[idx].topk(5).indices)
        str_res += ", ".join(similar_words) + "\n"
    modified_sentence = tokenizer.convert_ids_to_tokens(torch.argmax(mlm_pred, dim=-1))
    original_sentence = tokenizer.convert_ids_to_tokens(
        tokenizer(
            sentence[0],
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"].squeeze()
    )
    for i, _ in eq_class:
        original_sentence[i] = modified_sentence[i]
    str_res += "New sentence with argmax words for each word explored: " + " ".join(
        original_sentence
    )

    if not os.path.exists(txt_out_dir):
        os.makedirs(txt_out_dir)
    fname = os.path.join(txt_out_dir, f"{iteration}-{str_pred}.txt")
    with open(fname, "w") as file:
        file.write(str_res)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-type", type=str, choices=["same", "diff"], required=True)
    parser.add_argument("--objective", type=str, choices=["cls", "mask"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--delta", type=float, default=9e-1)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--iter", type=int, default=100)
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
        args.model_name, mask_or_cls=args.objective
    )
    deactivate_dropout_layers(bert_model)

    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, "input-space-exploration", args.exp_name + "-" + str_time
    )

    for idx, txt in enumerate(txts):
        tokenized_input = bert_tokenizer(
            txt,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        )
        keep_constant = 0
        if args.objective == "mask":
            keep_constant = [
                i
                for i, el in enumerate(tokenized_input["input_ids"].squeeze())
                if el == bert_tokenizer.mask_token_id
            ][0]
        w_ids = [
            i
            for i, el in enumerate(tokenized_input.word_ids())
            if el in eq_class_words[names[idx]]
        ]
        eq_class_word_ids = [
            i for i, el in enumerate(tokenized_input.word_ids()) if el in w_ids
        ]
        eq_class_words_and_ids[names[idx]] = {
            "keep_constant": (
                keep_constant,
                "[CLS]" if keep_constant == 0 else "[MASK]",
            ),
            "eq_class_w": [
                (ind, wrd)
                for ind, wrd in zip(eq_class_word_ids, eq_class_words[names[idx]])
            ],
        }

        embedded_input = bert_model.bert.embeddings(**tokenized_input)
        explore(
            same_equivalence_class=args.exp_type == "same",
            input_embedding=embedded_input,
            model=bert_model.bert.encoder,
            eq_class_emb_ids=(
                eq_class_word_ids if len(eq_class_word_ids) > 0 else None
            ),
            pred_id=keep_constant,
            device=device,
            delta=args.delta,
            threshold=args.threshold,
            n_iterations=args.iter,
            out_dir=os.path.join(res_path, names[idx]),
        )

    with torch.no_grad():
        for txt_dir in os.listdir(res_path):
            if os.path.isdir(os.path.join(res_path, txt_dir)):
                for filename in os.listdir(os.path.join(res_path, txt_dir)):
                    if os.path.isfile(
                        os.path.join(res_path, txt_dir, filename)
                    ) and filename.lower().endswith(".pkl"):
                        res = load_object(os.path.join(res_path, txt_dir, filename))
                        interpret(
                            sent_filename=(args.txt_dir, txt_dir),
                            model=bert_model,
                            decoder=(
                                bert_model.cls
                                if args.objective == "mask"
                                else bert_model.decoder
                            ),
                            tokenizer=bert_tokenizer,
                            class_map=class_map,
                            input_embedding=res["input_embedding"],
                            output_embedding=res["output_embedding"],
                            mask_or_cls=args.objective,
                            iteration=res["iteration"],
                            eq_class_words_ids=eq_class_words_and_ids[txt_dir],
                            txt_out_dir=os.path.join(
                                res_path, txt_dir, "interpretation"
                            ),
                        )


if __name__ == "__main__":
    main()
