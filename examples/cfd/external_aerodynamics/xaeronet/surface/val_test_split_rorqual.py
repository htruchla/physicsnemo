"""
SPLITTING TO TEST AND VALIDATION SET 
15% test 
15% val
"""
import os 
import json
import shutil
import random
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

def find_bin_files(directory:str)-> list[str]:
    """return a list that has absolute paths to .bin files in directory"""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".bin")
    )

def move_files(file_paths: list[str], destination_dir: str) -> list[str]:
    """move each file from file_path to destination_dir"""
    os.makedirs(destination_dir, exist_ok=True)
    new_paths = []
    for src in file_paths:
        filename = os.path.basename(src)
        dst = os.path.join(destination_dir, filename)
        shutil.move(src, dst)
        new_paths.append(dst)

    return new_paths

def save_manifest(file_paths: list[str], output_path: str, split_name: str)-> None:
    manifest = {
        "split": split_name,
        "num_samples": len(file_paths),
        "files": [os.path.basename(p) for p in file_paths],
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f" Manifest saved to {output_path}")

@hydra.main(version_base="1.3", config_path="conf",config_name="config")
def main(cfg: DictConfig) -> None:
    partitions_dir = to_absolute_path(cfg.partitions_path)
    val_dir = to_absolute_path(cfg.validation_partitions_path)
    test_dir = to_absolute_path(cfg.test_partitions_path)

    all_files = find_bin_files(partitions_dir)
    total = len(all_files)
    
    num_val = max(1, round(total * VAL_FRACTION))
    num_test = max(1, round(total * TEST_FRACTION))
    num_train = num_val - num_test

    if num_train <=0:
        raise RuntimeError(
            f"Dataset too smalll with a total of {num_train}, add more data or reduce fraction"
        )

    print(f"\nSplit plan:")
    print(f"  Train      : {num_train:>5}  ({num_train / total * 100:.1f}%)")
    print(f"  Validation : {num_val:>5}  ({num_val  / total * 100:.1f}%)")
    print(f"  Test       : {num_test:>5}  ({num_test / total * 100:.1f}%)")

    #split the data 

    seed = 42
    random.seed(seed)
    shuffled = all_files.copy()
    random.shuffle(shuffled)
    val_files = shuffled[:num_val]
    test_files = shuffled[num_val:num_val + num_test]
    train_files = shuffled[num_val + num_test:]

    #move the files
    moved_test = move_file(test_files, test_dir)
    moved_val = move_file(val_files, val_dir)

    #save the manifest 
    save_manifest(
        train_files,
        os.path.join(partitions_dir, "train_manifest.json"),
        "train",
    )

    save_manifest(
        moved_val,
        os.path.join(val_dir, "val_manifest.json"),
        "validation",
    )

    save_manifest(
        moved_test,
        os.path.join(val_dir, "test_manifest.json"),
        "test",
    )


if __name__ =="__main__":
    main()