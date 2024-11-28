import os 
import exifread, rawpy
from torchvision.transforms.v2 import Resize
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import tzip
import json, jsonlines

camera_crop_factor_dict = {
            # Full Frame
            'Canon Canon EOS 5D Mark IV': 1.0,
            'Canon Canon EOS 5D Mark III': 1.0,
            'Canon Canon EOS 5D Mark II': 1.0,
            'Canon Canon EOS 5D': 1.0,
            'Canon Canon EOS 6D': 1.0,
            'Canon Canon EOS 6D Mark II': 1.0,
            'Canon Canon EOS 5DS R': 1.0,
            'Canon Canon EOS 5DS': 1.0,
            'Canon Canon EOS-1D X': 1.0,
            'Canon Canon EOS-1D X Mark II': 1.0,
            'Canon Canon EOS-1Ds Mark III': 1.0,
            'Canon Canon EOS-1Ds Mark II'   : 1.0,
            'Canon Canon EOS R': 1.0,
            # APS-H
            'Canon Canon EOS-1D Mark II N': 1.26,
            'Canon Canon EOS-1D Mark II'  : 1.26,
            'Canon Canon EOS-1D Mark III': 1.26,
            'KODAK DCS460D         FILE VERSION 3': 1.3,
            # APS-C (Canon)
            'Canon Canon EOS 70D' : 1.6,
            'Canon Canon EOS 40D' : 1.62,
            'Canon Canon EOS 650D' : 1.61,
            'Canon Canon EOS 60D' : 1.61,
            'Canon Canon EOS 550D' : 1.61,
            'Canon Canon EOS 350D DIGITAL' : 1.62,
            'Canon Canon EOS 7D' : 1.61,
            'Canon Canon EOS 750D' : 1.61,
            'Canon Canon EOS 600D' : 1.61,
            'Canon Canon EOS 1200D' : 1.61,
            'Canon Canon EOS 20D' : 1.6,
            'Canon Canon EOS 200D' : 1.61,
            'Canon Canon EOS 500D' : 1.61,
            'Canon Canon EOS 100D' : 1.61,
            'Canon Canon EOS 700D' : 1.61,
            'Canon Canon EOS 450D' : 1.62,
            'Canon Canon EOS 30D' : 1.6,
            'Canon Canon EOS REBEL T3': 1.62,
            'Canon Canon EOS REBEL T5i': 1.61,
            'Canon Canon EOS REBEL T2i': 1.61,
            'Canon Canon EOS DIGITAL REBEL XSi': 1.62,
            'Canon Canon EOS D60' : 1.59,
            'Canon Canon EOS D30' : 1.59,
            'Canon Canon EOS 10D' : 1.59,
            'Canon Canon EOS DIGITAL REBEL XT': 1.62,
            'Canon Canon EOS DIGITAL REBEL XTi': 1.62,
            'Canon Canon EOS DIGITAL REBEL': 1.59,
            # APS-C (Nikon)
            'NIKON CORPORATION NIKON D40X': 1.52,
            'NIKON CORPORATION NIKON D70': 1.53,
            # APS-C Fukifilm
            'FUJIFILM FinePixS2Pro': 1.56,
            # 4/3-type
            'Panasonic DC-GH5' : 2.0,
            # 1/1.8" 
            'NIKON E990': 4.87,
            'Canon Canon PowerShot S70': 4.87,
            # 1/1.65" 
            # https://www.digicamdb.com/specs/leica_d-lux-3/
            'LEICA D-LUX 3': 4.47,
            # 1/1.7"
            'FUJIFILM FinePix F700': 4.6,
            'Canon Canon PowerShot G9': 4.6,
            'Canon Canon PowerShot G10': 4.6,
            # Leica Camera AG M8 Digital Camera
            'Leica Camera AG M8 Digital Camera': 1.33,

        }

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Curate CameraSettings20k')
    parser.add_argument('--source_dir', type=str, 
                        help='The root directory of the dataset')
    parser.add_argument('--target_dir', type=str,  
                        help='The target directory of the dataset')
    parser.add_argument('--image_size', type=int, default=1024, 
                        help='The size of the image')
    return parser.parse_args()

def read_raw_image(image_path, transform):
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, half_size=True)
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        image = transform(image)
    return image, H, W

def read_exif_from_dataset(RAISE_raw_image_dir='RAISE_raw',
                           DDPD_raw_image_dir= 'DDPD_raw',
                           PPR10K_raw_image_dir='PPR10K_raw',
                           PPR10K_tif_image_dir='PPR10K_360_tif',
                           ):
    
    exifs = []
    exif_jsons = []
    raw_image_files = []

    print('Start reading exif data from the raw image datasets...')
    print('Reading RAISE dataset...')
    dataset_path = os.path.join(args.source_dir, RAISE_raw_image_dir)
    pbar = tqdm(os.listdir(dataset_path))
    for raw_image_filename in pbar:
        pbar.set_description(f'Processing {raw_image_filename}')
        image_path = os.path.join(dataset_path, raw_image_filename)
        exif, exif_json = read_exif(image_path)
        if check_image_whether_having_camera_settings(exif):
            exifs.append(exif)
            exif_jsons.append(exif_json)
            raw_image_files.append(image_path)

    print('Reading DDPD dataset...')
    dataset_path = os.path.join(args.source_dir, DDPD_raw_image_dir)
    pbar = tqdm(os.listdir(dataset_path))
    for raw_image_filename in pbar:
        pbar.set_description(f'Processing {raw_image_filename}')
        image_path = os.path.join(dataset_path, raw_image_filename)
        exif, exif_json = read_exif(image_path)
        if check_image_whether_having_camera_settings(exif):
            exifs.append(exif)
            exif_jsons.append(exif_json)
            raw_image_files.append(image_path)

    print('Reading PPR10K dataset...')
    dataset_path = os.path.join(args.source_dir, PPR10K_raw_image_dir)
    pbar = tqdm(os.listdir(dataset_path))
    for raw_image_filename in pbar:
        pbar.set_description(f'Processing {raw_image_filename}')
        image_path = os.path.join(dataset_path, raw_image_filename)
        exif, exif_json = read_exif(image_path)

        if check_image_whether_having_camera_settings(exif):
            exifs.append(exif)
            exif_jsons.append(exif_json)
            raw_image_files.append(image_path)

        else:
            print('Can not extract exif data from the raw image, try to read from tiff image...')
            _, file_extension = os.path.splitext(raw_image_filename)
            tif_image_path = os.path.join(args.source_dir, PPR10K_tif_image_dir, raw_image_filename.replace(file_extension, '.tif'))
            print('reading...', tif_image_path)
            exif, exif_json = read_exif(tif_image_path)

            if check_image_whether_having_camera_settings(exif):
                exifs.append(exif)
                exif_jsons.append(exif_json)
                raw_image_files.append(image_path)
            else:
                print('Can not extract exif data from the tiff image...')
                print('Skip this image...')

    return exifs, exif_jsons, raw_image_files

def read_exif(file_path):
    with open(file_path, 'rb') as f:
        exif_tags = exifread.process_file(f, details=False)
        if len(exif_tags) == 0:
            return None, None
        # barrowed from https://github.com/rom1504/img2dataset/blob/a70e10d352ec11fd611b86ab81a29223a16c841e/img2dataset/downloader.py#L305-L311
        exif_json_string = json.dumps(
            {
                k: str(v).strip()
                for k, v in exif_tags.items()
                if v is not None
            }
        )

        exif = dict()
        for key, value in exif_tags.items():
            top_tag = key.split(' ')[0]
            new_key = key.split(' ')[1]
            if top_tag == 'Thumbnail' or (top_tag == 'Image' and new_key  == 'Tag')\
                or (top_tag == 'EXIF' and new_key  == 'DateTimeOriginal') \
                or top_tag == 'IFD':
                continue
            if new_key not in exif.keys():
                exif[new_key] = str(value)
            elif value != exif[new_key]:
                print('comflict!!')
                print(key, value)
                print(exif[new_key])
        return exif, exif_json_string

def check_image_whether_having_camera_settings(exif):
    if exif is None:
        return False
    else:
        HaveExposureTime = 'ExposureTime' in exif.keys()
        HaveFocalLength = 'FocalLength' in exif.keys()
        HaveFNumber = 'FNumber' in exif.keys()
        HaveISOSpeedRatings = 'ISOSpeedRatings' in exif.keys()
        if HaveFocalLength and HaveFNumber and HaveISOSpeedRatings and HaveExposureTime:
            if exif['FocalLength'] == '0' or exif['FNumber'] == '0' or exif['FocalLength'] == '65535' or exif['FNumber'] == '65535':
                return False
            return True
        else:
            return False

def get_camera_settings(exif):
    camera_settings = dict()
    HaveFocalLength = 'FocalLength' in exif.keys()
    HaveFocalLengthIn35mmFilm = 'FocalLengthIn35mmFilm' in exif.keys()
    if (not HaveFocalLengthIn35mmFilm) and HaveFocalLength:
        if "Make" in exif.keys() and "Model" in exif.keys():
            camera_name = f'{exif["Make"]} {exif["Model"]}'
        elif "Model" in exif.keys():
            camera_name = f'{exif["Model"]}'
        elif "Make" in exif.keys():
            camera_name = f'{exif["Make"]}'
        else:
            camera_name = None
        if camera_name in camera_crop_factor_dict.keys():
            crop_factor = camera_crop_factor_dict[camera_name]
            if '/' in exif['FocalLength']:
                camera_settings['FocalLengthIn35mmFilm'] = crop_factor * float(int(exif['FocalLength'].split('/')[0]) / \
                                                                int(exif['FocalLength'].split('/')[1]))
            else:
                camera_settings['FocalLengthIn35mmFilm'] = crop_factor * float(exif['FocalLength'])
        else:
            raise f'Undefined camera: {camera_name}'
        
    elif HaveFocalLengthIn35mmFilm and HaveFocalLength:
        if float(exif['FocalLength'])==0:
            if "Make" in exif.keys() and "Model" in exif.keys():
                camera_name = f'{exif["Make"]} {exif["Model"]}'
            elif "Model" in exif.keys():
                camera_name = f'{exif["Model"]}'
            elif "Make" in exif.keys():
                camera_name = f'{exif["Make"]}'
            else:
                camera_name = None
            if camera_name == 'SONY ILCE-7RM2':
                crop_factor = 1
            else:
                raise f'Undefined camera: {camera_name}'
        else:
            crop_factor = float(exif['FocalLengthIn35mmFilm']) / float(exif['FocalLength'])

        camera_settings['FocalLengthIn35mmFilm'] = float(exif['FocalLengthIn35mmFilm'])       

    if '/' in exif['FNumber']:
        camera_settings['FNumber'] = float(int(exif['FNumber'].split('/')[0]) / \
                                            int(exif['FNumber'].split('/')[1])) 

        camera_settings['FNumberIn35mmFilm'] = crop_factor * camera_settings['FNumber']
        
    else:
        camera_settings['FNumberIn35mmFilm'] = crop_factor * float(exif['FNumber'])

    if '/' in exif['ExposureTime']:
        camera_settings['ExposureTime'] = float(int(exif['ExposureTime'].split('/')[0]) / \
                                                int(exif['ExposureTime'].split('/')[1]))
    else:
        camera_settings['ExposureTime'] = float(exif['ExposureTime'])

    camera_settings['ISOSpeedRatingsIn35mmFilm'] = (crop_factor**2) * float(exif['ISOSpeedRatings'])

    camera_settings['CropFactor'] = crop_factor

    return camera_settings

if __name__ == '__main__':
    args = arg_parser()
    transform = Resize(args.image_size)
    
    exifs, exif_jsons, raw_image_files = read_exif_from_dataset()

    save_image_dir = os.path.join(args.target_dir, 'train')
    os.makedirs(save_image_dir, exist_ok=True)
    metadata_path = os.path.join(args.target_dir, 'train', 'metadata.jsonl')

    print('Start converting raw image to png image...')
    for exif, exif_jsons, raw_image_file_path in tzip(exifs, exif_jsons, raw_image_files):
        image, H, W = read_raw_image(raw_image_file_path, transform)
        file_name = os.path.splitext(os.path.basename(raw_image_file_path))[0]+'.png'
        
        image_metadata = get_camera_settings(exif)
        image_metadata['original_height'] = H
        image_metadata['original_width'] = W
        image_metadata['file_name'] = file_name
        image_metadata['exif'] = exif_jsons

        image.save(os.path.join(save_image_dir, 
                                file_name))

        with jsonlines.open(metadata_path, mode='a') as writer:
            writer.write(image_metadata)
            

