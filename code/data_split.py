# Some parts of this code is borrowed from https://github.com/xahidbuffon/FUnIE-GAN
import argparse,glob,json,os
import numpy as np

# Create data splits
def datasplit(image_path, prefix=""):
    paths = glob.glob(f"{image_path}/*.*")
    base_names = list(map(os.path.basename, paths))
    base_names = np.array(base_names)
    total_len = len(base_names)
    print(f"Total {total_len} data")
    valid_len = int(total_len * 0.1)
    indices = np.random.permutation(total_len)
    train_indices = indices[valid_len:]
    valid_indices = indices[:valid_len]

    train_names = base_names[train_indices].tolist()
    valid_names = base_names[valid_indices].tolist()
    train_names = list(map(lambda x: f"{prefix}{x}", train_names))
    valid_names = list(map(lambda x: f"{prefix}{x}", valid_names))
    print(f"Train has {len(train_names)} data from {image_path}")
    print(f"Valid has {len(valid_names)} data from {image_path}")
    return {"train": train_names, "valid": valid_names}


# Create data splits for unpaired
def unpaired_data(data_path):
    dt_splits = datasplit(f"{data_path}/trainA", prefix="trainA/")
    eh_splits = datasplit(f"{data_path}/trainB", prefix="trainB/")
    splits = dict()
    for key in ["train", "valid"]:
        splits[key] = dt_splits[key] + eh_splits[key]
    return splits

# Create data splits for paired
def paired_data(data_path):
    splits = datasplit(f"{data_path}/trainA")
    for key in ["train", "valid"]:
        paths = map(lambda x: f"{data_path}/trainB/{x}", splits[key])
        assert all(map(lambda x: os.path.isfile(x), paths))
    return splits




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Splitting")
    parser.add_argument("-d", "--data", default="../../EUVP_Dataset/Paired/underwater_dark", type=str, metavar="PATH",
                        help="path to data (default: none)")
    parser.add_argument("-p", "--pair", action="store_true",
                        help="Use this if data in pairs")

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    if args.pair:
        print("Paired dataset:")
        splits = paired_data(args.data)
    else:
        print("Unpaired dataset:")
        splits = unpaired_data(args.data)
    output_path = f"{args.data}/splits.json"
    with open(output_path, "w") as f:
        json.dump(splits, f)
    print(f"We write data splits to {output_path}")
