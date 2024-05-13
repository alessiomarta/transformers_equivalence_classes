import argparse
import json
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, type=str)
    parser.add_argument("--new", required=True, type=str)
    args = parser.parse_args()

    if not os.path.exists(args.new):
        os.makedirs(args.new)

    for outer_folder in os.listdir(args.old):

        inner_folder = os.path.join(args.old, outer_folder, "interpretation")

        for file in os.listdir(inner_folder):

            full_path = os.path.join(inner_folder, file)

            if file.endswith(".json"):

                new_path = os.path.join(args.new, f"{outer_folder}.json")

                with open(full_path, "r") as fin:
                    res = json.load(fin)

                with open(new_path, "w") as fout:
                    json.dump(res, fout)


if __name__ == "__main__":
    main()