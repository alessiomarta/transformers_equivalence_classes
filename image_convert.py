from torchvision import datasets, transforms
import torch
import os
import json
from collections import defaultdict
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import argparse
import gc
from experiments.experiments_utils import *
from analysis.vit_attention_exp import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--sample", type=int, default = 200)

    args = parser.parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    return args


if __name__ == "__main__":

    args = parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_sample_size = args.sample
    data_dir = f"./{args.dataset.lower()}_data"
    base_dir = f"./{args.dataset.lower()}_imgs"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    model, _ = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=torch.device(args.device),
    )

    testset = getattr(datasets, args.dataset.upper())(
        root=data_dir,
        download=True,
        train=False,
        transform = transform
    )

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        subset = torch.utils.data.Subset(testset, indices)
        data_test = subset.__getitems__(list(range(test_sample_size)))
        X_test = torch.stack(list(map(lambda tup: tup[0], data_test)))
        y_test = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), data_test)))
    else:
        X_test = testset.data
        y_test = testset.targets

    norm = Normalize(vmin=0, vmax=1)

    for y,label in enumerate(testset.classes):

        # Set saving directory
        label_dir = os.path.join(base_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Set config file
        configs = defaultdict(dict)

        # Filter by label
        y_indices = indices[y_test == y]
        X = X_test[y_test == y].to(torch.float32)

        for i in range(X.shape[0]):
            fname = f"img_{y_indices[i]}"
            print(fname)

            # Save image
            image = X[i,:,:]
            _, ax = plt.subplots()
            ax.tick_params(which = "both", bottom = False, top = False, left = False, right = False, labelbottom = False, labelleft = False)
            ax.imshow(image.squeeze().numpy().transpose((1,2,0)), cmap="gray", norm=norm)
            plt.savefig(os.path.join(label_dir, fname))
            plt.close()

            # Compute and save configs
            V_patches = image.shape[0] // model.patcher.patch_size
            H_patches = image.shape[1] // model.patcher.patch_size
            tot_patches = H_patches * V_patches
            N_patches = [1, tot_patches // 4, tot_patches // 2, 3*(tot_patches//4), tot_patches]
            x = image.unsqueeze(0)
            if x.dim() == 3:
                x = x.unsqueeze(0)

            transformer_attribution = generate_relevance(model, x).detach()
            patches_attribution = transformer_attribution.reshape((V_patches, -1, H_patches)).cpu().numpy().mean(axis = 1)
            sorted_idx = np.argsort(patches_attribution, axis = None)[::-1]

            for n in N_patches:
                if n == tot_patches:
                    configs[n][fname] = []
                else:
                    configs[n][fname] = sorted_idx[:n].tolist()

            del patches_attribution
            del transformer_attribution
            gc.collect()

        for n,config in configs.items():
            with open(os.path.join(label_dir, f"config_{n}.json"), "w") as f:
                json.dump(config, f)



    