import argparse
import torch
from bert_explain.bert_grad_rollout import BERTAttentionGradRollout
from experiments_utils import *
import json


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--txt-dir", type = str, required = True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ).type
    return args


def main():

    args = parse_args()
    model_name = args.model
    device = torch.device(args.device)
    txt_dir = args.txt_dir
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load texts
    txts, names = load_raw_sents(txt_dir)

    bert_tokenizer, bert_model = load_bert_model(
        model_name, mask_or_cls="cls"
    )

    deactivate_dropout_layers(bert_model)

    grad_rollout = BERTAttentionGradRollout(bert_model, discard_ratio=0.9)

    for name, sent in zip(names, txts):

        fname = os.path.join(out_dir, f"{name}.json")

        print("Document:", name)

        encoded_input = bert_tokenizer(sent, return_tensors = "pt")
        pred = bert_model(**encoded_input)
        pred = pred.logits.detach().numpy().flatten().argmax()

        mask = grad_rollout(encoded_input, category_index = pred)

        sentence_tokens = bert_tokenizer.tokenize(sent)

        output = {
            "tokens_imp": tuple(zip(mask.tolist(), sentence_tokens))
        }

        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(output, f)


if __name__ == "__main__":

    main()