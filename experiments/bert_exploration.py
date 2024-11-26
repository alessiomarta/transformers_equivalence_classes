"""
A module for exploring the input space of BERT models. It includes functionalities to visualize
the impact of individual tokens and their equivalence classes on the model's predictions, 
and to experiment with different configurations and equivalence classes.
"""

import argparse
import os
import logging as log
import time
from transformers import logging
import torch
from simec.logics import explore
from experiments.experiments_utils import (
    load_bert_model,
    deactivate_dropout_layers,
    load_raw_sents,
    save_object,
    load_json,
    ExplorationException,
)


# Configure the logger
log.basicConfig(
    filename="error_log.txt",
    filemode="a",
    level=log.ERROR,
    format="\n\n%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Directory containing data, config.json, and parameters.json. Automatically created with prepare_experiment.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory where to store the exploration output from this experiment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Where to run this experiment",
    )
    parser.add_argument("--cap-ex", default=True)

    arguments = parser.parse_args()
    if arguments.device is None:
        arguments.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ).type
    else:
        arguments.device = torch.device(arguments.device).type

    return arguments


def main():
    args = parse_args()
    device = torch.device(args.device)
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    txts, names = load_raw_sents(args.experiment_path)
    logging.set_verbosity_error()
    bert_tokenizer, bert_model = load_bert_model(
        params["model_path"], mask_or_cls=params["objective"], device=device
    )
    deactivate_dropout_layers(bert_model)
    bert_model = bert_model.to(device)

    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, os.path.basename(args.experiment_path) + "-" + str_time
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    sentence_embeddings = []
    if params["objective"] == "mlm":
        # in this case, we need to go one by one because we replace token with [MASK]
        for idx, txt in enumerate(txts):
            tokenized_input = bert_tokenizer(
                txt,
                return_tensors="pt",
                return_attention_mask=False,
                add_special_tokens=False if txt.strip().startswith("[CLS]") else True,
                padding="max_length",
            )
            tokenized_input["input_ids"][0][
                [config[names[idx]]["objective"]]
            ] = bert_tokenizer.mask_token_id
            sentence_embeddings.append(
                bert_model.bert.embeddings(**tokenized_input.to(device)).to(device)
            )
    else:  # we can go in parallel
        tokenized_input = bert_tokenizer(
            txts,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False if txts[0].startswith("[CLS]") else True,
            padding="max_length",
        )
        sentence_embeddings = (
            bert_model.bert.embeddings(**tokenized_input.to(device))
            .unsqueeze(1)
            .to(device)
        )

    algorithms = ["simec"]
    if params["algo"] == "both":
        algorithms.append("simexp")
    elif params["algo"] == "simexp":
        algorithms[0] = "simexp"
    for algorithm in algorithms:
        print(f"\t{algorithm.upper()} exploration phase")
        for idx, sentence_embedding in enumerate(sentence_embeddings):
            if params["objective"] == "mlm":
                if (
                    isinstance(config[names[idx]]["objective"], list)
                    and len(config[names[idx]]["objective"]) > 1
                ):
                    print("Skipping sentence because objective has more than 1 token.")
                    continue
            for r in range(params["repeat"]):
                print(
                    f"Sentence:{names[idx]}\t{idx+1}/{len(txts)}\tRepetition: {r+1}/{params['repeat']}"
                )
                try:
                    explore(
                        same_equivalence_class=algorithm == "simec",
                        input_embedding=sentence_embedding,
                        model=bert_model.bert.encoder,
                        eq_class_emb_ids=(
                            None
                            if config[names[idx]]["explore"] == []
                            else config[names[idx]]["explore"]
                        ),
                        pred_id=config[names[idx]]["objective"],
                        device=device,
                        threshold=params["threshold"],
                        delta_multiplier=params["delta_mult"],
                        n_iterations=params["iterations"],
                        out_dir=os.path.join(
                            res_path,
                            f"{algorithm}-{names[idx].split('.')[0]}-{str(r+1)}",
                        ),
                        keep_timing=True,
                        save_each=params["save_each"],
                        capping=args.experiment_path if args.cap_ex else "",
                    )
                except Exception as e:
                    log.error(
                        "Unhandled exception during exploration:\nContext:\n"
                        "res_path: %s\nalgorithm: %s\nname: %s\nparams_repeat: %d\nError: %s",
                        res_path,
                        algorithm,
                        names[idx],
                        r + 1,
                        e,
                        exc_info=True,
                    )
                    raise ExplorationException from e


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if not isinstance(e, ExplorationException):
            log.error("An error occurred: %s", e, exc_info=True)
        raise
