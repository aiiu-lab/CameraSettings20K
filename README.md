# CameraSettings20k
[[project page]](https://camera-settings-as-tokens.github.io/)[[demo]](https://huggingface.co/spaces/Camera-Settings-as-Tokens/Camera-Settings-as-Tokens)[[code]](https://github.com/aiiu-lab/Camera-Settings-as-Tokens)

![](preview.png)

The official data cuartion code for CameraSettings20k (from "Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models", SIGGRAPH Asia 2024).

## Requirements
We highly recommend using the [Conda](https://docs.anaconda.com/miniconda/) to build the environment. 

You can build and activate the environment by following commands. 
```bash
conda env create -f env.yml 
conda activate CameraSettings20k
```
After activating the environment, you need to install the requirements by running the following command due to [dependency issue](https://github.com/salesforce/LAVIS/issues/762) of LAVIS.
```bash
pip install opencv-python 
```

## Prepare Source Datasets
Please download the [RAISE](http://loki.disi.unitn.it/RAISE/), [DDPD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel), and [PPR10k](https://github.com/csjliang/PPR10K) datasets and put their raw images as follows. 
```
CameraSettings20k_src ┬ RAISE_raw 
                      ├ DDPD_raw # Put indoor and ourddoor CR2 images in the same folder
                      ├ PPR10k_raw # Put raw images ("PPR10K-dataset/raw") in this folder
                      └ PPR10K_360_tif # Put 360p tif images ("PPR10K-dataset/train_val_images_tif_360p") in this folder
```

## Data Curation

After preparing the source datasets, you can curate the CameraSettings20k (with out image caption) by running the following command. 
```bash
python data_curation.py --image_size <image_size> --source_dir <path to CameraSettings20k_src> --target_dir <path to save CameraSettings20k>
```
This code will generate the CameraSettings20k dataset with the following structure for training with [diffusers](https://huggingface.co/docs/diffusers/en/index) and [datasets](https://huggingface.co/docs/datasets/en/index). 
```
CameraSettings20k ┬ train ┬ metadata.jsonl
                          ├ <image_id_0>.png
                          ├ <image_id_1>.png
                          ├ ...
                          └ <image_id_n>.png
```

## Image Captioning with BLIP2

After curating the CameraSettings20k, you can generate the image caption by running the following command. 
```bash
python image_caption.py --dataset_dir <path to CameraSettings20k>
```

## Disclaimer
This dataset is for research purposes only. We do not own the rights to the images in the source datasets. Please refer to the original source datasets for the usage of the images.

## Citation
If you use CameraSettings20K, please cite our paper. 

```Bibtex
@inproceedings{fang2024camera,
  title={Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models},
  author={I-Sheng Fang and Yue-Hua Han and Jun-Cheng Chen},
  booktitle={SIGGRAPH Asia},
  year={2024}
}
```
