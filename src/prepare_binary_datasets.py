import shutil
from pathlib import Path


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "data" / "binary_datasets"


# ==========================================
# DATASET MAPPING
# ==========================================

DATASET_MAPPING = {
    "eyes": {
        "Open_Eyes": "open",
        "Closed_Eyes": "closed"
    },

    "mouth": {
        "Yawn": "yawn",
        "No_yawn": "no_yawn"
    }
}


SPLIT_MAPPING = {
    "Train": "train",
    "Val": "val",
    "Test": "test"
}


# ==========================================
# CLEAN OLD OUTPUT
# ==========================================

if OUTPUT_DIR.exists():

    print("Removing previous binary datasets...")

    shutil.rmtree(OUTPUT_DIR)


# ==========================================
# CREATE DATASETS
# ==========================================

print("\nCreating binary datasets...\n")


for dataset_name, class_mapping in DATASET_MAPPING.items():

    print(f"Creating {dataset_name} dataset...")


    for source_split, target_split in SPLIT_MAPPING.items():

        for source_class, target_class in class_mapping.items():

            source_path = (
                SOURCE_DIR
                / source_split
                / source_class
            )

            target_path = (
                OUTPUT_DIR
                / dataset_name
                / target_split
                / target_class
            )


            # Create destination folder

            target_path.mkdir(
                parents=True,
                exist_ok=True
            )


            # Copy images

            image_files = [
                file
                for file in source_path.iterdir()
                if file.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            ]


            for image_file in image_files:

                shutil.copy2(
                    image_file,
                    target_path / image_file.name
                )


            print(
                f"{target_split:5} | "
                f"{target_class:10} | "
                f"{len(image_files)} images"
            )


    print()


# ==========================================
# FINAL SUMMARY
# ==========================================

print("=" * 45)
print("BINARY DATASET PREPARATION COMPLETE")
print("=" * 45)

for dataset_name in DATASET_MAPPING:

    print(f"\n{dataset_name.upper()} DATASET")

    for split in ["train", "val", "test"]:

        split_path = (
            OUTPUT_DIR
            / dataset_name
            / split
        )

        print(f"\n{split.upper()}")

        for class_folder in sorted(
            split_path.iterdir()
        ):

            if class_folder.is_dir():

                count = len([
                    file
                    for file in class_folder.iterdir()
                    if file.suffix.lower()
                    in [".jpg", ".jpeg", ".png"]
                ])

                print(
                    f"{class_folder.name}: {count}"
                )