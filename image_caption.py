import os 
import torch   
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import tzip

import lavis

import json, jsonlines

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Captioning images of CameraSettings20k by BLIP2')
    parser.add_argument('--dataset_dir', type=str, 
                        help='The directory of the dataset')
    parser.add_argument('--blip_arch', type=str, default='blip2_opt',)
    parser.add_argument('--blip_model_type', type=str, default='caption_coco_opt6.7b')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = lavis.models.load_model_and_preprocess(
        name=args.blip_arch, model_type=args.blip_model_type, is_eval=True, device=device)
    
    metadata_path = os.path.join(args.dataset_dir, 'train', 'metadata.jsonl')
    metadata_backup_path = os.path.join(args.dataset_dir, 'train', 'metadata_backup.jsonl')
    if os.path.exists(metadata_path):
        os.rename(metadata_path, metadata_backup_path)
    else:
        if os.path.exists(metadata_backup_path):
            print(f'No metadata file found at {metadata_path}, but found backup metadata file at {metadata_backup_path}')
        else: 
            print(f'No metadata file found at {metadata_path} and {metadata_backup_path}')

    print('Start image captioning...')
    
    with jsonlines.open(metadata_backup_path) as reader:
        for image_metadata in tqdm(reader):
            file_name = image_metadata['file_name']
            image_path = os.path.join(args.dataset_dir, 'train', file_name)
            image = Image.open(image_path)
            with torch.no_grad():
                bllp_image = vis_processors["eval"](image).unsqueeze(0).to(device)
                caption = model.generate({"image": bllp_image})[0]
            image_metadata['text'] = caption
            with jsonlines.open(metadata_path, mode='a') as writer:
                writer.write(image_metadata)
