import torch
import numpy as np
import argparse
import json
from experiments_utils import *
from Transformer_Explainability.BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from Transformer_Explainability.BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification


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

    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name)

    explanations = Generator(bert_model)

    for name, sent in zip(names, txts):

        fname = os.path.join(out_dir, f"{name}.png")

        print("Document:", name)

        encoded_input = bert_tokenizer(sent, return_tensors = "pt")
        pred = bert_model(**encoded_input)
        pred = pred.logits.detach().numpy().flatten().argmax()

        expl = explanations.generate_LRP(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask, start_layer=0)[0]
        expl = (expl - expl.min()) / (expl.max() - expl.min())

        sentence_tokens = bert_tokenizer.tokenize(sent)

        output = {
            "tokens_imp": tuple(zip(expl.tolist(), sentence_tokens))
        }

        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(output, f)



if __name__ == "__main__":

    main()