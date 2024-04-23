import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import simec
from .utils import (
    load_bert_model,
    get_allowed_tokens,
)


def get_all_predictions(text_sentence, tokenizer, bert_model, closest_vectors=5):
    """Given a sentence with a masked token, yields the top closest_vectors predictions

    Args:
        text_sentence: A sentence with a mask token to be predicted
        closest_vectors: The number of possibile predictions we want, in decreasing order of probability. Defaults to 5.

    Returns:
        #closest_vectors predictions.
    """

    allowed_tokens = get_allowed_tokens(tokenizer)

    tokenized_input = tokenizer(
        text_sentence,
        return_tensors="pt",
        return_attention_mask=False,
        add_special_tokens=False,
    )
    mask_idx = [
        i
        for i, el in enumerate(tokenized_input["input_ids"].squeeze())
        if el == tokenizer.mask_token_id
    ]

    with torch.no_grad():
        predict = bert_model(**tokenized_input)[0]
    predict[0, mask_idx, allowed_tokens] = predict[0, mask_idx, allowed_tokens] * 100
    bert = tokenizer.convert_ids_to_tokens(
        predict[0, mask_idx].topk(closest_vectors).indices.squeeze()
    )

    return {"bert": bert}


def get_all_cls_predictions(text_sentence, tokenizer, bert_model, class_map):
    tokenized_input = tokenizer(
        text_sentence,
        return_tensors="pt",
        return_attention_mask=False,
        add_special_tokens=False,
    )

    with torch.no_grad():
        predict = bert_model(**tokenized_input)[0]
    return {"bert": class_map[torch.argmax(predict).item()]}


def save_colorbar(min_imp, max_imp, colormap, filename):
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=min_imp, vmax=max_imp)

    # Create a ScalarMappable and initialize a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Importance Level")

    # Save the colorbar as an image
    plt.savefig(filename)
    plt.close()


def generate_color_gradient(min_imp, max_imp, colormap):
    cmap = plt.get_cmap(colormap)
    gradient = [
        cmap(float(x) / (max_imp - min_imp))
        for x in np.linspace(min_imp, max_imp, num=256)
    ]
    return [
        f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
        for rgba in gradient
    ]


def create_html_with_highlighted_text(text, importance_levels, colors, colorbar_img):
    html_content = "<html><head><style>"
    # Map importance levels to the gradient indexes
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


# -----------------------------------------------------------------------------------------


def simec_bert(
    model,
    tokenizer,
    input_text,
    eq_class_words,
    mask_or_cls,
    device,
    class_map=None,
    delta=5.0,
    threshold=1e-2,
    num_iter=100,
    print_every_n_iter=10,
):
    """Build a polygonal approximating the equivalence class of a token given an embedded input.

    Args:
        encoder: The encoder part of the model.
        model_head: The prediction head of the model.
        delta: The lenght of the segment we proceed along in each iteration.
        threshold: The threshold parameter we use to separate null and and non-null eigenvalues.
          Below the threshold we consider an eigenvalue as null.
        num_iter: The number of iterations of the algorithm.
        embedded_input: The embedding of a sentence.
        eq_class_word_id: The id of the token of which we want to build the equivalence class.
        id_masked_word: The id of the masked word, which we want to keep constant.
        print_every_n_iter: The points built by the algorithm are printed every print_every_n_iter iterations.
    """

    # improvement: this could be batched, allowing for multiple sentences each time
    def pullback(input_simec, output_simec):
        # Compute the pullback metric
        jac = jacobian(output_simec, input_simec)
        jac_t = torch.transpose(jac, -2, -1)
        tmp = torch.bmm(jac, g)
        if device.type == "mps":
            # mps doen't support float64, must convert in float32
            pullback_metric = torch.bmm(tmp, jac_t).type(torch.float32)
        else:
            # The conversion to double is done in order to avoid the following error:
            # The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
            pullback_metric = torch.bmm(tmp, jac_t).type(torch.double)
        return torch.linalg.eigh(pullback_metric, UPLO="U")

    # Build the embedding of the sentence
    tokenized_input = tokenizer(
        input_text,
        return_tensors="pt",
        return_attention_mask=False,
        add_special_tokens=False,
    )
    if mask_or_cls == "mask":
        keep_constant = [
            i
            for i, el in enumerate(tokenized_input["input_ids"].squeeze())
            if el == tokenizer.mask_token_id
        ]  #!! only first item !!
    elif mask_or_cls == "cls":
        keep_constant = 0

    embedded_input = model.bert.embeddings(**tokenized_input)

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
    g = (
        torch.eye(model.config.hidden_size)
        .unsqueeze(0)
        .repeat(embedded_input.size(1), 1, 1)
        .to(device)
    )

    # Clone and require gradient of the embedded input
    emb_inp_simec = embedded_input.clone().to(device).requires_grad_(True)

    # Compute the output of the encoder. The output corresponding to the [MASK]
    # is what we want to keep constant
    encoder_output = model.bert.encoder(emb_inp_simec)[0].to(device)

    # Compute the pullback metric and its eigenvalues and eigenvectors
    eigenvalues, _ = pullback(
        output_simec=encoder_output[0, keep_constant].squeeze(),
        input_simec=emb_inp_simec,
    )

    max_eigenvalues = [
        torch.tensor(v).item() for v in torch.max(eigenvalues, dim=1).values.tolist()
    ]

    sentence = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"].squeeze())
    colorbar_img_path = "colorbar.png"

    # Generate a color gradient from the 'Blues' colormap
    colors = generate_color_gradient(
        min(max_eigenvalues), max(max_eigenvalues), "Oranges"
    )

    # Save colorbar image
    save_colorbar(
        min(max_eigenvalues), max(max_eigenvalues), "Oranges", colorbar_img_path
    )

    # Create HTML content
    html_content = create_html_with_highlighted_text(
        sentence, max_eigenvalues, colors, colorbar_img_path
    )

    # Write the HTML content to a file
    with open("highlighted_text.html", "w") as file:
        file.write(html_content)


def main():

    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_or_cls = "cls"
    class_map = None

    # Build the model
    model_name = "ctoraman/hate-speech-bert"
    # model_name = "bert-base"
    bert_tokenizer, bert_model = load_bert_model(model_name, mask_or_cls=mask_or_cls)

    # Input sentence
    input_text = "[CLS] Little stupid as bitch I don't fuck with yoooooouuuu.."
    # input_text = "[CLS] That nurse is a [MASK]"
    # Check predictions
    if mask_or_cls == "mask":
        prediction = get_all_predictions(
            input_text, bert_tokenizer, bert_model, closest_vectors=3
        )["bert"]
    elif mask_or_cls == "cls":
        class_map = {0: "Neutral", 1: "Offensive", 2: "Hate"}
        prediction = get_all_cls_predictions(
            input_text,
            bert_tokenizer,
            bert_model,
            class_map=class_map,
        )["bert"]
    print(prediction)
    print("---------------------------------------------------------------")

    # Run the algorithm
    simec_bert(
        model=bert_model,
        tokenizer=bert_tokenizer,
        input_text=input_text,
        eq_class_words=["i", "hate", "it", "!"],
        mask_or_cls=mask_or_cls,
        device=device,
        class_map=class_map,
    )


if __name__ == "__main__":
    main()
