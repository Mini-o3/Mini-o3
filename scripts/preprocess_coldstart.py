import os
import json
import argparse
import datasets
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

datasets.disable_caching()

def process_and_save(item, image_save_dir):
    idx = item.pop("idx")
    image_names = item.pop("image_names")
    images = item.pop("images")
    sample_save_dir = os.path.join(image_save_dir, f"minio3_coldstart_{idx}")
    os.makedirs(sample_save_dir, exist_ok=True)
    image_path_list = []
    for image, image_name in zip(images, image_names):
        image_save_path = os.path.join(sample_save_dir, image_name)
        image_path_list.append(image_save_path)
        assert isinstance(image, Image.Image)
        image.save(image_save_path)
    item["images"] = image_path_list
    item["conversations"] = eval(item["conversations"])
    for k in ["data_source", "sample_index", "rollout_index"]:
        item.pop(k, None)
    return item

def main(dataset_path, output_dir):
    image_save_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "Mini-o3-Coldstart.json")
    
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.add_column("idx", list(range(len(dataset))))

    processed_dataset = dataset.map(
        lambda item: process_and_save(item, image_save_dir),
        num_proc=64
    )

    json_data = processed_dataset.to_pandas().to_dict(orient="records")
    for item in tqdm(json_data):
        image_path_list = item['images'].tolist() if hasattr(item['images'], 'tolist') else list(item['images'])
        item["conversations"] = item["conversations"].tolist() if hasattr(item["conversations"], "tolist") else list(item["conversations"])
        item["images"] = [image_path['path'] if isinstance(image_path, dict) else str(image_path) for image_path in image_path_list]
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Save image and json_file to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Mini-o3 Coldstart Dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Absolute path to save processed images and json")
    args = parser.parse_args()
    main(args.dataset_path, args.output_dir)