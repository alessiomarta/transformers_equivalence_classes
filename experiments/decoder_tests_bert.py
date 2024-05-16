import argparse
import torch
import string
from transformers import BertTokenizerFast, logging
from experiments_utils import (
    load_bert_model,
    deactivate_dropout_layers,
    load_raw_sents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--objective", type=str, choices=["cls", "mask"], required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    txts, names = load_raw_sents(args.txt_dir)

    bert_tokenizer, bert_model = load_bert_model(
        args.model_name, mask_or_cls=args.objective, device=device
    )
    deactivate_dropout_layers(bert_model)
    bert_model = bert_model.to(device)
    decoder_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    decoder = bert_model.cls if args.objective == "mask" else bert_model.decoder

    mlm_errors = 0
    total = 0
    pred_errors = []
    punct_errors = 0
    first_token_errors = 0
    for idx, txt in zip(names, txts):
        if "[SEP]" not in txt:
            txt += " [SEP]"
        if "[CLS]" not in txt:
            txt = "[CLS] " + txt
        tokenized_input = bert_tokenizer(
            txt,
            return_tensors="pt",
            return_attention_mask=False,
            add_special_tokens=False,
        ).to(device)
        if args.objective == "cls":
            check = decoder_tokenizer(
                txt,
                return_tensors="pt",
                return_attention_mask=False,
                add_special_tokens=False,
            ).to(device)
            if not all(
                check["input_ids"].squeeze() == tokenized_input["input_ids"].squeeze()
            ):
                print("not equal tokenizations")

        # finding token of which to keep the prediction constant
        keep_constant = 0
        if args.objective == "mask":
            keep_constant = [
                i
                for i, el in enumerate(tokenized_input["input_ids"].squeeze())
                if el == bert_tokenizer.mask_token_id
            ][0]

        input_embedding = bert_model.bert.embeddings(**tokenized_input).to(device)
        original_pred = torch.argmax(bert_model(**tokenized_input)[0], dim=-1).squeeze()

        decoded_tokenized_ids = torch.argmax(
            (
                decoder(input_embedding)
                if args.objective == "cls"
                else decoder(bert_model.bert.encoder(input_embedding)[0])
            ).squeeze(),
            dim=-1,
        )
        decoded_tokenized_ids[0] = bert_tokenizer.cls_token_id
        decoded_tokenized_ids[-1] = bert_tokenizer.sep_token_id
        if keep_constant != 0:
            decoded_tokenized_ids[keep_constant] = bert_tokenizer.mask_token_id
        decoded_pred = torch.argmax(
            bert_model(
                input_ids=decoded_tokenized_ids.unsqueeze(0),
                token_type_ids=tokenized_input["token_type_ids"],
            )[0],
            dim=-1,
        ).squeeze()
        if (
            not all(
                tokenized_input["input_ids"].squeeze()[1:-1]
                == decoded_tokenized_ids[1:-1]
            )
            and False
        ):
            print("______________________")
            print(
                f"\tOriginal:\n{bert_tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'].squeeze())}\n\tDecoded:\n{bert_tokenizer.convert_ids_to_tokens(decoded_tokenized_ids)}"
            )
        first_token_err = False
        punct_err = False
        for idx, (ori, mod) in enumerate(
            zip(
                tokenized_input["input_ids"].squeeze()[1:-1],
                decoded_tokenized_ids[1:-1],
            )
        ):
            total += 1
            if ori != mod and ori != bert_tokenizer.mask_token_id:
                mlm_errors += 1
                if mod == 1012 and idx == 0:
                    first_token_errors += 1
                    first_token_err = True
                if (
                    bert_tokenizer.convert_ids_to_tokens([mod])[0] in string.punctuation
                    and bert_tokenizer.convert_ids_to_tokens([ori])[0]
                    in string.punctuation
                ):
                    punct_errors += 1
                    punct_err = True
        if args.objective == "mask":
            if original_pred[keep_constant] != decoded_pred[keep_constant]:
                pred_errors.append((first_token_err, punct_err))
        else:
            if original_pred != decoded_pred:
                pred_errors.append((first_token_err, punct_err))

    print(f"Total MLM errors: {mlm_errors}/{total} tokens")
    print(f"\t of which {punct_errors} are mismatched punctuations")
    print(
        f"\t of which {first_token_errors} are first tokens after [CLS] predicted as ."
    )
    print(f"Total prediction errors: {len(pred_errors)}/{len(txts)} sentences")
    print(
        f"\t{len([x for x in pred_errors if x[0]])} of which contain first token errors"
    )
    print(
        f"\t{len([x for x in pred_errors if x[1]])} of which contain punctuation errors"
    )


if __name__ == "__main__":
    main()
