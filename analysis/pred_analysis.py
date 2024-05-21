import os
import argparse
import json
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import pipeline
from models.vit import ViTForClassification
import torch


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_model(model_path: str, config_path: str, device: torch.device):
    if device.type == "cpu":
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(model_path)
    config = load_config(config_path)
    model = ViTForClassification(config)
    model.load_state_dict(checkpoint)
    return model.to(device), config


def load_raw_images(img_dir: str):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    images = []
    images_names = []
    for filename in os.listdir(img_dir):
        if os.path.isfile(
            os.path.join(img_dir, filename)
        ) and filename.lower().endswith(image_extensions):
            image = Image.open(os.path.join(img_dir, filename)).convert("L")
            if image.size != (28, 28):
                image = image.resize((28, 28))
            images.append(transform(image))
            images_names.append(filename.split(".")[0])
    return torch.stack(images), images_names


def load_raw_sents(txt_dir: str):
    txts = []
    txts_names = []
    for filename in os.listdir(txt_dir):
        if os.path.isfile(
            os.path.join(txt_dir, filename)
        ) and filename.lower().endswith(".txt"):
            with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as file:
                txts.append(file.readlines()[0])
            txts_names.append(filename.split(".")[0])
    return txts, txts_names


def collect_data_and_preds(res_dir, objective, original_preds=None, pipe=None):
    if pipe:
        data = {
            "file-name": [],
            "original-sentence": [],
            "mod-sentence": [],
            "original-pred": [],
            "original-pred-proba": [],
            "algorithm": [],
            "iteration": [],
            "explore-token": [],
            "alternative-token": [],
            "alternative-token-proba": [],
            "modified-pred-proba": [],
            "alternative-pred": [],
            "alternative-pred-proba": [],
        }
    else:
        data = {
            "file-name": [],
            "original-pred": [],
            "algorithm": [],
            "iteration": [],
            "modified-pred-proba": [],
            "alternative-pred": [],
            "modified-original-proba": [],
        }
    selected_dirs = [
        d
        for d in os.listdir(res_dir)
        if objective in d and os.path.isdir(os.path.join(res_dir, d))
    ]
    for exp_dir in selected_dirs:
        for res in tqdm(os.listdir(os.path.join(res_dir, exp_dir)), desc=exp_dir):
            if os.path.isdir(os.path.join(res_dir, exp_dir, res)):
                files = [
                    filename
                    for filename in os.listdir(
                        os.path.join(res_dir, exp_dir, res, "interpretation")
                    )
                    if os.path.isfile(
                        os.path.join(res_dir, exp_dir, res, "interpretation", filename)
                    )
                    and filename.lower().endswith("-stats.json")
                ]
                for j_file in tqdm(files, desc=res):
                    stats = json.load(
                        open(
                            os.path.join(
                                res_dir, exp_dir, res, "interpretation", j_file
                            ),
                            "r",
                        )
                    )
                    if pipe:
                        eq_class_wrds_keys = [
                            k
                            for k in stats.keys()
                            if "cap-probas-" in k
                            and "pre-cap-probas-" not in k
                            and k != "cap-probas-mod"
                        ]
                        for word in eq_class_wrds_keys:
                            for alternative in stats[word]:
                                data["original-sentence"].append(
                                    " ".join(stats["original-sentence"][1:-1])
                                )
                                data["algorithm"].append(exp_dir.split("-")[0])
                                data["iteration"].append(int(j_file.split("-")[0]))
                                data["explore-token"].append(word.split("-")[-1])
                                data["alternative-token"].append(alternative[0])
                                data["alternative-token-proba"].append(alternative[1])
                                # Compute modified sentence prediction
                                word_index = stats["original-sentence"].index(
                                    word.split("-")[-1]
                                )
                                mod_sentence = " ".join(
                                    (
                                        stats["original-sentence"][1:word_index]
                                        + [alternative[0]]
                                        + stats["original-sentence"][
                                            word_index + 1 : -1
                                        ]
                                    )
                                )
                                data["mod-sentence"].append(mod_sentence)
                                data["file-name"].append(res)
                                alternative_pred_max = pipe(mod_sentence)[0]
                                if objective == "msk":
                                    data["original-pred"].append(
                                        original_preds[res]["token_str"]
                                    )
                                    alternative_pred = pipe(
                                        mod_sentence,
                                        targets=[original_preds[res]["token_str"]],
                                    )[0]
                                    data["alternative-pred"].append(
                                        alternative_pred_max["token_str"]
                                    )
                                else:
                                    data["original-pred"].append(
                                        original_preds[res]["label"]
                                    )
                                    alternative_pred = pipe(
                                        mod_sentence, return_all_scores=True
                                    )[0]
                                    alternative_pred = [
                                        d
                                        for d in alternative_pred
                                        if d["label"] == original_preds[res]["label"]
                                    ][0]
                                    data["alternative-pred"].append(
                                        alternative_pred_max["label"]
                                    )
                                data["modified-pred-proba"].append(
                                    alternative_pred["score"]
                                )
                                data["original-pred-proba"].append(
                                    original_preds[res]["score"]
                                )
                                data["alternative-pred-proba"].append(
                                    alternative_pred_max["score"]
                                )
                    else:
                        data["file-name"].append(res)
                        data["original-pred"].append(int(stats["original_image_pred"]))
                        data["modified-pred-proba"].append(
                            stats["modified_image_pred_proba"]
                        )
                        data["modified-original-proba"].append(
                            stats["modified_original_pred_proba"]
                        )
                        data["alternative-pred"].append(
                            int(stats["modified_image_pred"])
                        )
                        data["algorithm"].append(exp_dir.split("-")[0])
                        data["iteration"].append(int(j_file.split("-")[0]))
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", type=str, required=True)
    parser.add_argument("--cls-dir", type=str, required=True)
    parser.add_argument("--mlm-dir", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--vit-model-path", type=str, required=True)
    parser.add_argument("--vit-config-path", type=str, required=True)
    parser.add_argument("--plots-out-dir", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    res_path = args.res_dir
    print("Loading models...")
    # models
    cls_pipe = pipeline("text-classification", model="ctoraman/hate-speech-bert")
    msk_pipe = pipeline("fill-mask", model="bert-base-uncased")
    vit_model, _ = load_model(
        model_path=args.vit_model_path,
        config_path=args.vit_config_path,
        device=device,
    )
    vit_model = vit_model.to(device)
    vit_model.eval()

    print("Loading data...")
    # data
    msk_txts, msk_names = load_raw_sents(args.mlm_dir)
    msk_txts = [sent.replace("[CLS]", "").replace("[SEP]", "") for sent in msk_txts]
    msk_preds = {k: p[0] for k, p in zip(msk_names, msk_pipe(msk_txts))}
    cls_txts, cls_names = load_raw_sents(args.cls_dir)
    cls_txts = [sent.replace("[CLS]", "").replace("[SEP]", "") for sent in cls_txts]
    cls_preds = {k: p for k, p in zip(cls_names, cls_pipe(cls_txts))}

    palette = {"simec": "cornflowerblue", "simexp": "darkorange"}
    style = {"simec": "", "simexp": (3, 2)}
    if not os.path.exists(args.plots_out_dir):
        os.makedirs(args.plots_out_dir)

    # experiments
    print("ViT experiment...")
    data_vit = pd.DataFrame.from_dict(collect_data_and_preds(res_path, "vit"))
    data_vit = data_vit.loc[data_vit["file-name"] != "img_34"]
    data_vit["same-eq-class"] = (
        data_vit["original-pred"] == data_vit["alternative-pred"]
    )
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
    sns.lineplot(
        data=data_vit.loc[data_vit["same-eq-class"] == False],
        ax=ax[0],
        x="iteration",
        y="modified-pred-proba",
        style="algorithm",
        hue="algorithm",
        palette=palette,
        dashes=style,
    )
    ax[0].set(
        ylabel="Probability (pre-Softmax)",
        title="Equivalence class probability value for $\mathbf{y} \in Y_c$\nViT for Digit Classification",
        xlabel="Iteration",
    )
    sns.lineplot(
        data=data_vit.loc[data_vit["same-eq-class"] == True],
        ax=ax[1],
        x="iteration",
        y="modified-pred-proba",
        style="algorithm",
        hue="algorithm",
        palette=palette,
        dashes=style,
    )
    ax[1].set(
        ylabel="Probability (pre-Softmax)",
        title="Equivalence class probability value for $\mathbf{y} \in Y_s$\nViT for Digit Classification",
        xlabel="Iteration",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_out_dir, "eq-class-proba-vit.png"))
    plt.close()

    print("BERT MSK experiment...")
    data_msk = pd.DataFrame.from_dict(
        collect_data_and_preds(res_path, "msk", msk_preds, msk_pipe)
    )
    data_msk["same-eq-class"] = (
        data_msk["original-pred"] == data_msk["alternative-pred"]
    )
    print("Generating plot...")

    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
    sns.lineplot(
        data=data_msk.loc[data_msk["same-eq-class"] == False],
        ax=ax[0],
        x="iteration",
        y="modified-pred-proba",
        style="algorithm",
        hue="algorithm",
        palette=palette,
        dashes=style,
    )
    ax[0].set(
        ylabel="Probability",
        title="Equivalence class probability value for $\mathbf{y} \in Y_c$\nBERT for Masked Language Modeling",
        xlabel="Iteration",
    )
    sns.lineplot(
        data=data_msk.loc[data_msk["same-eq-class"] == True],
        ax=ax[1],
        x="iteration",
        y="modified-pred-proba",
        style="algorithm",
        hue="algorithm",
        palette=palette,
        dashes=style,
    )
    ax[1].set(
        ylabel="Probability",
        title="Equivalence class probability value for $\mathbf{y} \in Y_s$\nBERT for Masked Language Modeling",
        xlabel="Iteration",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_out_dir, "eq-class-proba-bert-msk.png"))
    plt.close()

    print("BERT CLS experiment...")
    data_cls = pd.DataFrame.from_dict(
        collect_data_and_preds(res_path, "cls", cls_preds, cls_pipe)
    )
    data_cls = data_cls.loc[data_cls["file-name"] != "sentence_2"]
    data_cls["same-eq-class"] = (
        data_cls["original-pred"] == data_cls["alternative-pred"]
    )
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
    sns.lineplot(
        data=data_cls.loc[data_cls["same-eq-class"] == False],
        ax=ax[0],
        x="iteration",
        y="modified-pred-proba",
        style="algorithm",
        hue="algorithm",
        palette=palette,
        dashes=style,
    )
    ax[0].set(
        ylabel="Probability",
        title="Equivalence class probability value for $\mathbf{y} \in Y_c$\nBERT for Hate Speech Detection",
        xlabel="Iteration",
    )
    sns.lineplot(
        data=data_cls.loc[data_cls["same-eq-class"] == True],
        ax=ax[1],
        x="iteration",
        y="modified-pred-proba",
        style="algorithm",
        hue="algorithm",
        palette=palette,
        dashes=style,
    )
    ax[1].set(
        ylabel="Probability",
        title="Equivalence class probability value for $\mathbf{y} \in Y_s$\nBERT for Hate Speech Detection",
        xlabel="Iteration",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_out_dir, "eq-class-proba-bert-cls.png"))
    plt.close()


if __name__ == "__main__":
    main()
