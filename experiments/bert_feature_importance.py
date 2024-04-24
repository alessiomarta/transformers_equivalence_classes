"""
A module for analyzing feature importance in BERT models. This includes functionalities
to highlight text based on importance levels, save these visualizations, and perform
interpretations based on eigenvalues derived from the model's embeddings.
"""

import os
import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from transformers import BertTokenizerFast
from simec.logics import pullback_eigenvalues
from experiments_utils import (
    load_bert_model,
    get_allowed_tokens,
    deactivate_dropout_layers,
    load_raw_sents,
    load_raw_sent,
    load_object,
)


def save_colorbar(min_imp: float, max_imp: float, colormap: str, filename: str) -> None:
    """
    Save a color gradient bar as an image file.

    Args:
        min_imp: Minimum importance level for color mapping.
        max_imp: Maximum importance level for color mapping.
        colormap: The name of the matplotlib colormap to use.
        filename: Path to save the colorbar image.
    """
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=min_imp, vmax=max_imp)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Importance Level")

    plt.savefig(filename)
    plt.close()


def generate_color_gradient(min_imp: float, max_imp: float, colormap: str) -> list:
    """
    Generate a list of color values in RGBA format as a gradient from minimum to maximum importance levels.

    Args:
        min_imp: Minimum importance level for color mapping.
        max_imp: Maximum importance level for color mapping.
        colormap: The name of the matplotlib colormap to use.

    Returns:
        List of colors in RGBA format.
    """
    cmap = plt.get_cmap(colormap)
    gradient = [
        cmap(float(x) / (max_imp - min_imp))
        for x in np.linspace(min_imp, max_imp, num=256)
    ]
    return [
        f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
        for rgba in gradient
    ]


def create_html_with_highlighted_text(
    text: list, importance_levels: list, colors: list, colorbar_img: str
) -> str:
    """
    Create an HTML document with text highlighted based on importance levels.

    Args:
        text: List of words to highlight.
        importance_levels: Corresponding importance levels for each word.
        colors: List of colors for each importance level.
        colorbar_img: Path to the colorbar image to include in the HTML.

    Returns:
        HTML content as a string.
    """
    html_content = "<html><head><style>"
    color_index = {
        level: int(
            (level - min(importance_levels))
            / (max(importance_levels) - min(importance_levels))
            * 255
        )
        for level in importance_levels
    }
    for i, level in enumerate(importance_levels):
        color = colors[color_index[level]]
        html_content += f".imp{i} {{ background-color: {color}; padding: 0 2px; border-radius:10%;font-family: 'Trebuchet MS', sans-serif;}}\n"
    html_content += "</style></head><body><p>"
    for i, word in enumerate(text):
        html_content += f'<span class="imp{i}">{word}</span> '
    html_content += f'</p><img src="{colorbar_img}" alt="Colorbar" style="width:50%; height:auto;"><body></html>'
    return html_content


def interpret(
    sent_filename: tuple,
    model: torch.nn.Module,
    input_embedding: torch.Tensor,
    decoder: callable,
    tokenizer: BertTokenizerFast,
    eigenvalues: torch.Tensor,
    output_embedding: torch.Tensor,
    keep_constant: int,
    mask_or_cls: str,
    class_map: dict = None,
    txt_out_dir: str = ".",
) -> None:
    """
    Perform interpretation of text based on model embeddings and eigenvalues, output results as HTML.

    Args:
        sent_filename: Tuple containing directory and sentence filename.
        model: Loaded BERT model.
        input_embedding: Input embeddings tensor.
        decoder: Function or module to decode predictions.
        tokenizer: Tokenizer for the model.
        eigenvalues: Eigenvalues for each token's contribution to the embedding.
        output_embedding: Output embeddings tensor.
        keep_constant: Index to keep constant during interpretations.
        mask_or_cls: Specify whether 'mask' for masked LM or 'cls' for classification.
        class_map: Optional dictionary mapping class indices to names.
        txt_out_dir: Directory to save the output HTML file.
    """
    txts_dir, sent_name = sent_filename
    sentence, _ = load_raw_sent(txts_dir, sent_name)
    sentence_tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(
            sentence,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"].squeeze()
    )
    max_eigenvalues = [
        torch.tensor(v).item() for v in torch.max(eigenvalues, dim=1).values.tolist()
    ]

    colors = generate_color_gradient(
        min(max_eigenvalues), max(max_eigenvalues), "Oranges"
    )

    if not os.path.exists(txt_out_dir):
        os.makedirs(txt_out_dir)

    save_colorbar(
        min(max_eigenvalues),
        max(max_eigenvalues),
        "Oranges",
        os.path.join(txt_out_dir, "colorbar.png"),
    )
    html_content = create_html_with_highlighted_text(
        sentence_tokens, max_eigenvalues, colors, "colorbar.png"
    )

    allowed_tokens = get_allowed_tokens(tokenizer)

    if mask_or_cls == "mask":
        mlm_pred = decoder(output_embedding)[0]
        mlm_pred[:, allowed_tokens] = mlm_pred[:, allowed_tokens] * 100
        str_pred = tokenizer.convert_ids_to_tokens(
            [torch.argmax(mlm_pred[keep_constant]).item()]
        )[0]
    else:
        mlm_pred = decoder(input_embedding)[0]
        cls_pred = model.classifier(model.bert.pooler(output_embedding))
        str_pred = class_map[torch.argmax(cls_pred).item()]

    fname = os.path.join(txt_out_dir, f"{str_pred}.html")
    with open(fname, "w") as file:
        file.write(html_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, choices=["cls", "mask"], required=True)
    parser.add_argument("--exp-name", type=str, required=True)
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
    keep_constant_dict = eq_class_words.copy()
    class_map = None
    if args.objective == "cls":
        class_map = {int(k): v for k, v in eq_class_words["class-map"].items()}

    bert_tokenizer, bert_model = load_bert_model(
        args.model_name, mask_or_cls=args.objective
    )
    deactivate_dropout_layers(bert_model)

    str_time = time.strftime("%Y%m%d-%H%M%S")
    res_path = os.path.join(
        args.out_dir, "feature-importance", args.exp_name + "-" + str_time
    )

    for idx, txt in enumerate(txts):
        tokenized_input = bert_tokenizer(
            txt,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        )
        keep_constant = 0  # Adjust based on model architecture and use case
        if args.objective == "mask":
            keep_constant = [
                i
                for i, el in enumerate(tokenized_input["input_ids"].squeeze())
                if el == bert_tokenizer.mask_token_id
            ][0]
        keep_constant_dict[names[idx]] = keep_constant
        embedded_input = bert_model.bert.embeddings(**tokenized_input)

        pullback_eigenvalues(
            input_embedding=embedded_input,
            model=bert_model.bert.encoder,
            pred_id=keep_constant,
            device=device,
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
                            eigenvalues=res["eigenvalues"],
                            tokenizer=bert_tokenizer,
                            class_map=class_map,
                            input_embedding=res["input_embedding"],
                            output_embedding=res["output_embedding"],
                            mask_or_cls=args.objective,
                            keep_constant=keep_constant_dict[txt_dir],
                            txt_out_dir=os.path.join(
                                res_path, txt_dir, "interpretation"
                            ),
                        )


if __name__ == "__main__":
    main()
