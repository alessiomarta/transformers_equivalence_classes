from utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--device", type=str)

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
    model_path = args.model_path
    config_path = args.config_path
    device = torch.device(args.device)

    # MNIST data
    trainloader, _ = prepare_data(test=False)

    model, _ = load_model(
        model_path=model_path,
        config_path=config_path,
        device=device,
    )

    deactivate_dropout_layers(model)

    # Get a mini-batch of train data loaders
    imgs, _ = next(iter(trainloader))
    imgs = imgs.to(device)
    # take first image keeping batch dimension
    img = imgs[0].unsqueeze(0)