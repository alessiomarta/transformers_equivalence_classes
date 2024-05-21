import argparse
from models.vit import *
from experiments_utils import *
import pickle
import torch
import pandas as pd
import os

def parse_args() -> argparse.Namespace:

    # python vit_predictions.py --exp-dir ../res/mnist_experiment/input-space-exploration/mnist_simec-20240520-165902/img_932 --model-path ../saved_models/vit/model_20.pt --config-path ../saved_models/vit/config.json --out-dir ../res/mnist_experiment/simec_probs --device cpu

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, type=str)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


def main():

    args = parse_args()
    device = args.device
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=torch.device(device),
    )
    deactivate_dropout_layers(model)
    model = model.to(device)

    for folder in os.listdir(args.exp_dir):

        if "." in folder:
            continue 

        probs = []

        for file in os.listdir(os.path.join(args.exp_dir, folder)):
            if file.endswith(".pkl"):
                with open(os.path.join(args.exp_dir, folder, file), "rb") as f:
                    d = pickle.load(f)
                x = d['output_embedding']

                y = model.classifier(x[:, 0]).reshape(-1)

                probs.append(y.detach().cpu().numpy().tolist())

        df = pd.DataFrame(probs)
        df.to_csv(os.path.join(args.out_dir, f"{folder}.csv"))

        preds = df.idxmax(axis = 1)
        first_pred = preds.values[0]
        N_first = preds.value_counts()[first_pred]
        print("Same as first prediction:", N_first)


if __name__ == "__main__":
    main()

    
