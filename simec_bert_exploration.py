import torch
from jacobian_function import jacobian
from utils import load_bert_model, get_allowed_tokens


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
        jac = jacobian(output_simec, input_simec)[eq_class_word_ids]
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
    no_split_tokens = []
    for split_w in tokenizer.convert_ids_to_tokens(
        tokenized_input["input_ids"].squeeze()
    ):
        if split_w[:2] == "##":
            no_split_tokens[-1] += split_w[2:]
        else:
            no_split_tokens.append(split_w)
    w_ids = [i for i, el in enumerate(no_split_tokens) if el in eq_class_words]
    eq_class_word_ids = [
        i for i, el in enumerate(tokenized_input.word_ids()) if el in w_ids
    ]

    embedded_input = model.bert.embeddings(**tokenized_input)

    # Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
    g = (
        torch.eye(model.config.hidden_size)
        .unsqueeze(0)
        .repeat(len(eq_class_word_ids), 1, 1)
        .to(device)
    )

    # Clone and require gradient of the embedded input
    emb_inp_simec = embedded_input.clone().to(device).requires_grad_(True)

    # Compute the output of the encoder. The output corresponding to the [MASK]
    # is what we want to keep constant
    encoder_output = model.bert.encoder(emb_inp_simec)[0].to(device)

    allowed_tokens = get_allowed_tokens(tokenizer)

    # Keep track of the length of the polygonal
    distance = torch.zeros(len(eq_class_word_ids))
    for i in range(num_iter):
        # Compute the pullback metric and its eigenvalues and eigenvectors
        eigenvalues, eigenvectors = pullback(
            output_simec=encoder_output[0, keep_constant].squeeze(),
            input_simec=emb_inp_simec,
        )

        # Select a random eigenvectors corresponding to a null eigenvalue.
        # We consider an eigenvalue null if it is below a threshold value-
        number_null_eigenvalues = torch.count_nonzero(eigenvalues < threshold, dim=1)
        null_vecs, zero_eigenvals = [], []
        for emb in range(eigenvalues.size(0)):
            if number_null_eigenvalues[emb]:
                id_eigen = torch.randint(0, number_null_eigenvalues[emb], (1,)).item()
                null_vecs.append(eigenvectors[emb, :, id_eigen].type(torch.float))
                zero_eigenvals.append(eigenvalues[emb, id_eigen].type(torch.float))
            else:
                null_vecs.append(torch.zeros(1).type(torch.float))
                zero_eigenvals.append(torch.zeros(1).type(torch.float))
        null_vecs = torch.stack(null_vecs, dim=0)
        zero_eigenvals = torch.stack(zero_eigenvals, dim=0)

        with torch.no_grad():
            # Proceeed along a null direction
            emb_inp_simec[0, eq_class_word_ids] = (
                emb_inp_simec[0, eq_class_word_ids] + delta * null_vecs
            )
            distance += zero_eigenvals * delta

            if i % print_every_n_iter == 0:
                tmp = encoder_output.cpu()
                if mask_or_cls == "mask":
                    pred = model.cls(tmp)[0]
                    pred[:, allowed_tokens] = pred[:, allowed_tokens] * 100
                elif mask_or_cls == "cls":
                    pred = model.decoder.cls(
                        model.decoder.bert.encoder(emb_inp_simec)[0]
                    )[0]
                for idx, w in zip(eq_class_word_ids, eq_class_words):
                    print(w.upper())
                    similar_word = tokenizer.convert_ids_to_tokens(
                        pred[idx].topk(5).indices
                    )
                    print(pred[idx])
                    print("First five words in the equivalence class:")
                    print(similar_word)
                print("Length of the polygonal in the embedding space :", distance)
                if mask_or_cls == "mask":
                    print(
                        "Argmax of the output:",
                        torch.argmax(pred[keep_constant]).item(),
                    )
                    print(
                        "Max of the output:",
                        torch.max(pred[keep_constant]).item() / 100,
                    )
                    print(
                        "Output token:",
                        tokenizer.convert_ids_to_tokens(
                            [torch.argmax(pred[keep_constant]).item()],
                        ),
                    )
                elif mask_or_cls == "cls":
                    print("Whole sentence with argmax for each token")
                    print(
                        " ".join(
                            tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1))[
                                1:
                            ]
                        )
                    )
                    pred = model.classifier(model.bert.pooler(tmp))
                    print(
                        "Argmax of the output:",
                        torch.argmax(pred).item(),
                    )
                    print(
                        "Max of the output:",
                        torch.max(pred).item() / 100,
                    )
                    print(
                        "Output class:",
                        class_map[torch.argmax(pred).item()],
                    )
                print("---------------------------------------------------------------")

        # Prepare for next iteration
        emb_inp_simec = emb_inp_simec.to(device).requires_grad_(True)
        encoder_output = model.bert.encoder(emb_inp_simec)[0].to(device)


# -----------------------------------------------------------------------------------------


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
    input_text = "[CLS] I hate it!"
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
