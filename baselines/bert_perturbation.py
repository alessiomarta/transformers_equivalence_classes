import argparse
import torch
import numpy as np
import time
import os
import json
from transformers import BertForMaskedLM
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa as sdpa_mask
import sys
sys.path.append("../experiments")
from simec.logics import OutputOnlyModel, jacobian
from experiments_utils import (
    load_raw_sents,
    deactivate_dropout_layers,
    load_json,
    load_object,
    load_bert_model,
)
from tqdm import tqdm, trange
import gc

# Objective: 
# 1. travelling long distances by maintaining invariance (SIMEC)
# 2. travelling short distances to find other equivalence classes (SIMEXP)

EPS= 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Directory containing data, config.json, and parameters.json. Automatically created with prepare_experiment.py",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="How many iteration to run the experiment.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Directory where to store the exploration output from this experiment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Where to run this experiment",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default = 2,
        help="Batch size.",
    )

    arguments = parser.parse_args()
    if arguments.device is None:
        arguments.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        arguments.device = torch.device(arguments.device)

    return arguments


def orthogonal_tensor(v):

    batch_size, seq_len, embedding_size = v.shape
    random_dim = np.random.randint(low = 0, high=embedding_size)
    all_dims = list(range(v.shape[-1]))
    all_dims.remove(random_dim)
    new_v  = v.clone()
    new_v[..., random_dim] = -1 * torch.sum(new_v[..., all_dims], dim = -1)
    new_v = new_v.view(batch_size, seq_len, embedding_size)

    return new_v


def gaussian_orthogonal_noise(v, std):

    noise = torch.normal(mean=torch.zeros_like(v), std=std.diagonal() + EPS)
    ort_direction = torch.nn.functional.normalize(orthogonal_tensor(v))
    
    assert ort_direction.dim() == 3
    magnitude = torch.bmm(ort_direction, noise.transpose(-1, -2)[..., [0]]) # index 0 just to take the first computed random noise vector

    return magnitude * ort_direction


def random_explore(model, data_loader, min_embs, max_embs, device, n_iterations, explore_patches, objectives, step_size = 0.1, direction ="same"):

    model = OutputOnlyModel(model)
    multiplier = 1 if direction == "same" else -1
    results = []
    objectives = [[ob] for ob in objectives]
    step_size = multiplier * step_size

    for idx, input_batch in enumerate(data_loader):

        embeddings = input_batch[0].clone().to(device)
        attention_mask = input_batch[1].to(device)
        extended_attention_mask = sdpa_mask(attention_mask, embeddings.dtype, tgt_len = embeddings.shape[1])

        if isinstance(model.model, BertForMaskedLM):
            output_embedding = model.model.bert.encoder(embeddings, attention_mask = extended_attention_mask)
            indices = model.model.cls(output_embedding['last_hidden_state']).argsort(dim = -1)[:,:,-10:]
        else:
            indices = None

        sigma = torch.abs(
            torch.cov(input_batch[0].view(-1, input_batch[0].shape[-1]).T)
        ).sqrt().to(device)

        with torch.no_grad():
            original_labels = model(embeddings, attention_mask = extended_attention_mask)[1]
            original_labels = original_labels.flatten()

        embeddings.requires_grad_(True)
        batch_size = embeddings.shape[0]
        embedding_size = embeddings.shape[-1]

        for t in trange(n_iterations, desc=f"Batch {idx}"):
            
            results.append([])
            gradients, _ = jacobian(embeddings, model, pred_id = objectives, select = indices)
            # gradients.shape = (batch_size, output_classes, N_tokens+1, embedding_size)

            with torch.no_grad():
                probits = model(embeddings, attention_mask = extended_attention_mask)[0].squeeze().detach().cpu().reshape(batch_size, -1, gradients.shape[1])
                original_and_top_class_probs = [
                    {
                        "batch_idx": idx,
                        "original": probits[j, objectives[j], original_labels[j]].tolist(), 
                        "top": [v.flatten().max().item() for v in probits[j, objectives[j]]],
                        "iteration": t
                    }
                for j in range(batch_size)]
                results[idx].extend(original_and_top_class_probs)

                for exp_patch in explore_patches:
                    embeddings[:, exp_patch, :] = embeddings[:, exp_patch, :] + step_size * gaussian_orthogonal_noise(gradients[:, 0, exp_patch, :].reshape(batch_size, -1, embedding_size), std = sigma)

                # Project back into the feasible region
                embeddings = torch.clamp(embeddings, min_embs.to(device), max_embs.to(device))
                del probits, gradients

        del embeddings
        torch.cuda.empty_cache()
        gc.collect()

    return results



def main():

    args = parse_args()
    if args.experiment_path.endswith("/"):
        args.experiment_path = args.experiment_path[:-1]
    params = load_json(os.path.join(args.experiment_path, "parameters.json"))
    config = load_json(os.path.join(args.experiment_path, "config.json"))
    
    if args.out_dir is None:
        raise ValueError("No output path specified in out-dir argument.")
    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, os.path.basename(args.experiment_path) + "-" + str_time
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    n_iterations = params["iterations"]

    device = torch.device(args.device)
    txts, names = load_raw_sents(args.experiment_path)
    batch_size = args.batch_size

    tokenizer, model = load_bert_model(
        params["model_path"], mask_or_cls=params["objective"], device=device
    )
    deactivate_dropout_layers(model)
    model = model.to(device) 

    #quando l'obiettivo è mlm, bisogna mettere la maschera al posto del token da tenere costante (objective)
    if params["objective"] in ["mlm", "msk", "mask"]:
        for i, (t, n) in enumerate(zip(txts, names)):
            t = tokenizer.tokenize(t)
            obj = config[n]["objective"]
            t[obj] = tokenizer.mask_token
            txts[i] = " ".join(t).replace(" ##", "")

    # Tokenizing and embedding layers
    input_tokens = tokenizer(
        txts,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=False if params["objective"] in ["mlm", "msk", "mask"] else True, # nelle frasi hatespeech non ci sono [cls] e [sep]
        padding = True
    ).to(device) 
    attention_masks = input_tokens.pop("attention_mask").cpu().to(bool)
    token_embeddings = model.bert.embeddings(**input_tokens).cpu()
    token_embeddings = torch.utils.data.TensorDataset(token_embeddings, attention_masks)
    sent_loader = torch.utils.data.DataLoader(
        token_embeddings,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    min_embs = load_object(os.path.join(args.experiment_path, "min_distribution.pkl"))
    max_embs = load_object(os.path.join(args.experiment_path, "max_distribution.pkl"))

    explore_patches = [config[name]["explore"] for name in names]
    objectives = [config[name]["objective"] for name in names]
    
    for alg in ["same", "opposite"]:
        stats = random_explore(model, sent_loader, min_embs, max_embs, device, n_iterations, explore_patches, objectives, step_size = 0.1, direction = alg)

        with open(os.path.join(res_path, f"perturbation_results_{alg}.json"), "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    main()
