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

    original_sent_ids = deepcopy(encoded_sent.ids)
    # Mask the objective token - already did it in loading sentences
    
    # Prepare the input embedding
    original_sentence = " ".join([tok for m,tok in zip(encoded_sent.special_tokens_mask, tokenizer.convert_ids_to_tokens(original_sent_ids)) if not m]).replace(" ##", "")
    # original_proba.shape = (n_iterations, max_len, vocab_size) OR (n_iterations, output_size)
    model.eval()
    with torch.no_grad():
        input_embedding = model.bert.embeddings(torch.tensor(encoded_sent.ids).unsqueeze(0))
        attention_mask = torch.tensor(encoded_sent.attention_mask).unsqueeze(0).to(device)
        extended_attention_mask = sdpa_mask(attention_mask, input_embedding.dtype, tgt_len = input_embedding.shape[1])
        if extended_attention_mask is not None: # it is none if attention_mask is all 1
            extended_attention_mask = extended_attention_mask.to(device)
        
        sequence_output = decoder.bert.encoder(
            input_embedding,
            attention_mask = extended_attention_mask)[0]
        # pooled_output = decoder.bert.pooler(sequence_output) not used in BertForMaskedLM class
        mlm_preds = decoder.cls(sequence_output)
        # mlm_preds.shape = n_iterations, max_len, vocab_size
        maxima = torch.argmax(mlm_preds, dim=-1).squeeze()
        # Decoding and re-encoding
        original_tokenized = tokenizer.tokenize(original_sentence, add_special_tokens = True)
        decoded_texts = []
        wrongpunctcount = 0
        for i, w in enumerate(original_tokenized):
            converted = tokenizer.convert_ids_to_tokens([maxima[i]])[0]
            if not(w in ["[CLS]", "[SEP]", "[UNK]", "[MASK]"]):
                decoded_texts.append((int(converted == w), i, converted, w))
            if converted == "." and i in [1, len(original_tokenized)-2] and w != converted:
                wrongpunctcount +=1
        errors = sum(1-s[0]  for s in decoded_texts)
        accuracy = sum(s[0] for s in decoded_texts)/len(decoded_texts)
        if accuracy != 1.0:
            #print(decoded_texts)
            print("equivalence class:", [e[0] for e in eq_class])
            print("punct instead of first or last token: ", wrongpunctcount)
            print("tot errors:", errors)
            print("accuracy:", accuracy)
            topk_maxima =  torch.topk(mlm_preds, k=10, dim = -1).indices.squeeze()
            for err in [el for el in decoded_texts if el[0] == 0]:
                alternatives = tokenizer.convert_ids_to_tokens(topk_maxima[err[1]])
                print("V" if err[-1] in alternatives else "X","alternatives for", err, ":", alternatives)
            print("--------------------------------------------------------")
        



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

    tokenized = bert_tokenizer(
        txts,
        return_tensors="pt",
        return_attention_mask=False,
        add_special_tokens=False if params["objective"] in ["mlm", "msk", "mask"] else True,# nelle frasi hatespeech non ci sono [cls] e [sep]
        padding=True,
    )

    tokenized_texts = {name.split(".")[0]:encoded for name,encoded in zip(names, tokenized.encodings)}
    done = []
    for res_dir in os.listdir(args.results_dir):
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
            if name not in done:
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
                done.append(name)


if __name__ == "__main__":
    main()
