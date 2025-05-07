"""
A module for exploring the input space of BERT models. It includes functionalities to visualize
the impact of individual tokens and their equivalence classes on the model's predictions, 
and to experiment with different configurations and equivalence classes.

Example usage:
    To run this script, use the following command:
    
    python bert_exploration_interpretation.py --experiment-path /path/to/experiment --results-dir /path/to/results --device cuda
"""

import argparse
import os
import logging as log
from tqdm import tqdm
from transformers import BertTokenizerFast, logging
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa as sdpa_mask
from tokenizers import Encoding
import torch
from numpy import around
from copy import deepcopy
import sys
sys.path.append("./")
from typing import List
from experiments_utils import (
    load_raw_sents,
    load_raw_sent,
    deactivate_dropout_layers,
    load_bert_model,
    save_object,
    load_object,
    save_json,
    load_json,
    load_metadata_tensors
)

class InterpretationException(Exception):
    pass


# Configure the logger
log.basicConfig(
    filename="error_log.txt",
    filemode="a",
    level=log.ERROR,
    format="\n\n%(asctime)s - %(levelname)s - %(message)s",
)


def interpret(
    encoded_sent: Encoding,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    tokenizer: BertTokenizerFast,
    input_embedding: torch.Tensor,
    eq_class_words_ids: dict,
    mask_or_cls: str,
    iteration: List[int],
    device: torch.device,
    class_map: dict = None,
    path: str = "",
    top_k: int = 5
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

    eq_class, keep_constant = (
        eq_class_words_ids["eq_class_w"],
        eq_class_words_ids["keep_constant"],
    )
    eq_class_ids = [t[0] for t in eq_class]
    keep_constant_id, keep_constant_txt = keep_constant
    interpretation_path = os.path.join(path, "interpretation")
    os.makedirs(interpretation_path, exist_ok=True)

    original_sent_ids = deepcopy(encoded_sent.ids)
    # Mask the objective token - already did it in loading sentences
    '''
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    original_sent_ids[keep_constant_id] = mask_id if mask_or_cls.lower() in ['mask','msk','mlm'] else cls_id
    '''
    
    # Prepare the input embedding
    input_embedding = input_embedding.to(device, dtype = model.dtype)
    original_sentence = " ".join([tok for m,tok in zip(encoded_sent.special_tokens_mask, tokenizer.convert_ids_to_tokens(original_sent_ids)) if not m]).replace(" ##", "")
    # original_proba.shape = (n_iterations, max_len, vocab_size) OR (n_iterations, output_size)
    model.eval()
    with torch.no_grad():
        attention_mask = torch.tensor(encoded_sent.attention_mask).unsqueeze(0).to(device)
        extended_attention_mask = sdpa_mask(attention_mask, input_embedding[0].dtype, tgt_len = input_embedding[0].shape[0])
        if extended_attention_mask is not None: # it is none if attention_mask is all 1
            extended_attention_mask = extended_attention_mask.to(device)
        original_proba = model(
            input_ids = torch.tensor(original_sent_ids).unsqueeze(0).to(device),
            attention_mask = attention_mask # not passing through the encoder directly, attention is managed in the class
        ).logits.squeeze()
        
        if mask_or_cls.lower() in ['mask','msk','mlm']:
            mlm_preds = decoder(model.bert.encoder(
                input_embedding,
                attention_mask = extended_attention_mask)[0]) # these are logits!
            # mlm_preds.shape = n_iterations, max_len, vocab_size
            maxima = torch.argmax(mlm_preds, dim=-1)
            # maxima.shape = n_iterations, max_len

            # Original text stats
            original_proba_top_k_ids = torch.argsort(original_proba[keep_constant_id], descending = True).tolist()[:top_k]
            json_stats = {
                "original_sentence": original_sentence,
                "original_image_pred_proba": original_proba[keep_constant_id, original_proba_top_k_ids].to("cpu"),
                "original_image_pred_proba_tokens": original_proba_top_k_ids,
                "original_image_pred": original_proba_top_k_ids[0]
            }
            
            # Decoding and re-encoding
            maxima[:,keep_constant_id] = tokenizer.mask_token_id
            original_tokenized = tokenizer.tokenize(original_sentence, add_special_tokens = False)
            decoded_texts = []
            for row in maxima:
                decoded_text = []
                for i, w in enumerate(original_tokenized):
                    if i not in eq_class_ids or i == 0:
                        decoded_text.append(w)
                    else:
                        decoded_text.append(tokenizer.convert_ids_to_tokens([row[i]])[0])
                decoded_texts.append(" ".join(decoded_text).replace(" ##", ""))
                        
            re_encoded_input = tokenizer(decoded_texts, padding = True, return_tensors="pt", add_special_tokens=False)
            new_logits = model(**re_encoded_input.to(device)).logits
            # new_logits.shape = n_iterations, max_len, vocab_size

            # Iterations
            for i,t in enumerate(iteration):
                json_stats['embedding_pred_proba'] = mlm_preds[i,keep_constant_id,original_proba_top_k_ids].to("cpu")
                json_stats['embedding_pred'] = original_proba_top_k_ids[torch.argmax(mlm_preds[i,keep_constant_id,original_proba_top_k_ids]).to("cpu").item()]

                json_stats["modified_sentence"] = decoded_texts[i]
                json_stats['modified_patches'] = mlm_preds[i,eq_class_ids].argmax(dim=-1).to("cpu").tolist()

                json_stats['modified_image_pred_proba'] = new_logits[i,keep_constant_id,original_proba_top_k_ids].to("cpu")

                json_stats['modified_image_pred'] = original_proba_top_k_ids[torch.argmax(new_logits[i,keep_constant_id,original_proba_top_k_ids]).to("cpu").item()]

                json_stats['modified_original_pred_proba'] = new_logits[i,keep_constant_id,original_proba_top_k_ids[0]].to("cpu").item()

                save_object(json_stats, os.path.join(interpretation_path, f"{t}-stats.pkl"))
                with open(os.path.join(interpretation_path, f"{t}-{json_stats['original_image_pred']}-{json_stats['embedding_pred']}-{json_stats['modified_image_pred']}.txt"), "w", encoding="utf-8") as f:
                    f.write(json_stats['modified_sentence'])

        # Classification
        else:
            #decoder.bert.encoder(input_embedding,attention_mask = torch.tensor(encoded_sent.attention_mask).unsqueeze(0).to(device).to(bool))[0]
            sequence_output = decoder.bert.encoder(
                input_embedding,
                attention_mask = extended_attention_mask)[0]
            # pooled_output = decoder.bert.pooler(sequence_output) not used in BertForMaskedLM class
            mlm_preds = decoder.cls(sequence_output)
            # mlm_preds.shape = n_iterations, max_len, vocab_size
            maxima = torch.argmax(mlm_preds, dim=-1)
            # maxima.shape = n_iterations, max_len

            # Original text stats
            json_stats = {
                "original_sentence": original_sentence,
                "original_image_pred_proba": original_proba.to("cpu"),
                "original_image_pred": torch.argmax(original_proba).to("cpu")
            }

            # Predictions
            cls_preds = model.classifier(
                model.bert.pooler(
                    model.bert.encoder(
                        input_embedding,
                        attention_mask = extended_attention_mask)[0]
                )
            )
            # cls_preds.shape = n_iterations, output_size

            # Decoding and re-encoding
            original_tokenized = tokenizer.tokenize(original_sentence, add_special_tokens = True)
            decoded_texts = []
            for row in maxima:
                decoded_text = []
                for i, w in enumerate(original_tokenized):
                    if i not in eq_class_ids or i == 0:
                        decoded_text.append(w)
                    else:
                        decoded_text.append(tokenizer.convert_ids_to_tokens([row[i]])[0])
                decoded_texts.append(" ".join(decoded_text).replace(" ##", ""))
            
            re_encoded_input = tokenizer(decoded_texts, padding = True, return_tensors="pt", add_special_tokens=False)
            new_logits = model(**re_encoded_input.to(device)).logits
            # new_logits.shape = n_iterations, output_size

            # Iterations
            for i,t in enumerate(iteration):
                json_stats['embedding_pred_proba'] = cls_preds[i].to("cpu")
                json_stats['embedding_pred'] = torch.argmax(cls_preds[i]).to("cpu").item()

                json_stats["modified_sentence"] = decoded_texts[i]
                json_stats['modified_patches'] = mlm_preds[i,eq_class_ids].argmax(dim=-1).to("cpu").tolist()

                json_stats['modified_image_pred_proba'] = new_logits[i].to("cpu")

                json_stats['modified_image_pred'] = torch.argmax(new_logits[i]).to("cpu").item()

                json_stats['modified_original_pred_proba'] = new_logits[i,json_stats['original_image_pred']].to("cpu").item()

                save_object(json_stats, os.path.join(interpretation_path, f"{t}-stats.pkl"))
                with open(os.path.join(interpretation_path, f"{t}-{json_stats['original_image_pred']}-{json_stats['embedding_pred']}-{json_stats['modified_image_pred']}.txt"), "w", encoding="utf-8") as f:
                    f.write(json_stats['modified_sentence'])


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Directory containing input data, config.json, and parameters.json. Automatically created with prepare_experiment.py",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory where the exploration output from this experiment is stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Where to run this experiment",
    )
    parser.add_argument("--cap-ex", default=True)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    if args.experiment_path.endswith("/"):
        args.experiment_path = args.experiment_path[:-1]
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    device = torch.device(args.device)

    txts, names = load_raw_sents(args.experiment_path)
    class_map = None
    if params['objective'] == "cls":
        class_map = {0: "polite", 1:"neutral", 2:"hatespeech"} # non vedo class-map in questo json, faccio forzato
    logging.set_verbosity_error()
    bert_tokenizer, bert_model = load_bert_model(
        params['model_path'], mask_or_cls=params['objective'], device=device
    )
    deactivate_dropout_layers(bert_model)
    bert_model = bert_model.to(device)
    
    if params["objective"] in ["mlm", "msk", "mask"]:
        for i, (t, n) in enumerate(zip(txts, names)):
            t = bert_tokenizer.tokenize(t)
            obj = config[n]["objective"]
            t[obj] = bert_tokenizer.mask_token
            txts[i] = " ".join(t).replace(" ##", "") 

    tokenized_texts = bert_tokenizer(
        txts,
        return_tensors="pt",
        return_attention_mask=False,
        add_special_tokens=False if params["objective"] in ["mlm", "msk", "mask"] else True,# nelle frasi hatespeech non ci sono [cls] e [sep]
        padding=True,
    )

    tokenized_texts = {name.split(".")[0]:encoded for name,encoded in zip(names, tokenized_texts.encodings)}

    for res_dir in tqdm(os.listdir(args.results_dir), desc = "Iterating over result folders"):
        if res_dir.startswith("sime"):
            res_path = os.path.join(args.results_dir, res_dir)

            algo, name, rep = res_dir.split("-")

            keep_constant = config[name + ".txt"]['objective']

            # object to make interpretation easier
            sent_explore_and_obj = {
                "keep_constant": (
                    keep_constant,
                    "[CLS]" if keep_constant == 0 else config[name + ".txt"]['masked'],
                ),
                "eq_class_w": [
                    (ind, wrd)
                    for ind, wrd in zip(
                        config[name + ".txt"]['explore'], config[name + ".txt"]['tokens']
                    )
                ]
            }

            with torch.no_grad():
                
                output = load_metadata_tensors(res_path)

                interpret(
                    encoded_sent = tokenized_texts[name],
                    model = bert_model,
                    decoder = bert_model.decoder if hasattr(bert_model, "decoder") else bert_model.cls,
                    tokenizer = bert_tokenizer,
                    input_embedding = output['input_embedding'],
                    eq_class_words_ids = sent_explore_and_obj,
                    mask_or_cls = params['objective'],
                    iteration = output['iteration'],
                    device = device,
                    class_map = class_map,
                    path = res_path,
                    top_k = 5
                )


if __name__ == "__main__":
    main()
