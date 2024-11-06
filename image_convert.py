from torchvision import datasets, transforms
import torch
import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import argparse
import gc
import sys
sys.path.append("./experiments")
sys.path.append("./analysis")
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
        _, indices = train_test_split(list(range(len(testset))), test_size = test_sample_size, random_state=42, stratify=testset.targets)
        subset = torch.utils.data.Subset(testset, indices)
        data_test = subset.__getitems__(list(range(test_sample_size)))
        X_test = torch.stack(list(map(lambda tup: tup[0], data_test)))
        y_test = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), data_test)))
    else:
        indices = list(range(len(testset)))
        X_test = testset.data
        y_test = testset.targets

    norm = Normalize(vmin=0, vmax=1)
    indices = torch.tensor(indices)

    for y,label in enumerate(testset.classes):

        # Set saving directory
        label_dir = os.path.join(base_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Set config file
        configs = defaultdict(dict)
        attributions = defaultdict(dict)

        # Filter by label
        y_indices = indices[y_test == y]
        X = X_test[y_test == y].to(torch.float32)

        for i in range(X.shape[0]):
            fname = f"img_{y_indices[i]}"
            print(fname)

            # Save image
            image = X[i]
            _, ax = plt.subplots()
            ax.tick_params(which = "both", bottom = False, top = False, left = False, right = False, labelbottom = False, labelleft = False)
            ax.imshow(image.numpy().transpose((1,2,0)), cmap="gray", norm=norm)
            plt.savefig(os.path.join(label_dir, fname))
            plt.close()

            # Compute and save configs
            V_patches = image.shape[1] // model.patcher.patch_size
            H_patches = image.shape[2] // model.patcher.patch_size
            tot_patches = H_patches * V_patches
            N_patches = [1, tot_patches // 4, tot_patches // 2, 3*(tot_patches//4), tot_patches]
            perc = ["one", "q1", "q2", "q3", "all"]
            x = image.unsqueeze(0)
            if x.dim() == 3:
                x = x.unsqueeze(0)

            transformer_attribution = generate_relevance(model, x).detach()
            patches_attribution = transformer_attribution.reshape((V_patches, -1, H_patches)).cpu().numpy().mean(axis = 1)
            sorted_idx = np.argsort(patches_attribution, axis = None)[::-1]
            sorted_scores = np.sort(patches_attribution, axis = None)[::-1]

            for n,a in zip(N_patches, perc):
                if n == tot_patches:
                    configs[a][fname] = []
                    attributions[a][fname] = {
                        "idx": [],
                        "att": [],
                        "img_dim": list(image.shape),
                        "patch_dim": [model.patcher.patch_size, model.patcher.patch_size]
                    }
                else:
                    configs[a][fname] = sorted_idx[:n].tolist()
                    attributions[a][fname] = {
                        "idx": sorted_idx[:n].tolist(),
                        "att": sorted_scores[:n].tolist(),
                        "img_dim": list(image.shape),
                        "patch_dim": [model.patcher.patch_size, model.patcher.patch_size]
                    }

            del patches_attribution
            del transformer_attribution
            gc.collect()

        for n,config in configs.items():
            with open(os.path.join(label_dir, f"config_{n}.json"), "w") as f:
                json.dump(config, f)

        for n,config in attributions.items():
            with open(os.path.join(label_dir, f"attrib_{n}.json"), "w") as f:
                json.dump(config, f)

    