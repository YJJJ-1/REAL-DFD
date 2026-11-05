# asym_multi

This package follows the DeepfakeBench `training/` directory structure and includes only the entry points, registry, datasets, detectors, trainer, metrics code, and YAML files required for asym_multi.

## External Data and Environment

The package does not include code for other detection methods, datasets, dataset JSON files, auxiliary VGGFace2 images, pretrained CLIP models. 

- `./datasets/`
- `./preprocessing/dataset_json/`
- `./real-candidates/` (contains preprocessed VGGFace2 images)
- `./active_learning/clip-model/`

Run the package in an existing DeepfakeBench Python environment. The paths for data and logs in `training/config/train_config.yaml` and `test_config.yaml` have been set to absolute paths. When running on another machine, update these two files as well as the `pretrained` and `unpaired_real_dir` settings in the detector YAML file.

## Extraction and Training

```bash
unzip asym_multi.zip
cd asym_multi
bash run_train.sh
```

To specify eight GPUs, use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_train.sh`. The script uses `torchrun --nproc_per_node=8`. The current configuration trains on FF++ c23, uses 8/32 frames for training/testing respectively.


## Testing with the Best Checkpoint Only

Replace the path below with the directory of the actual new run. This command does not perform training; it only loads `test/avg/ckpt_best.pth` and prints evaluation metrics to the terminal:

```bash
cd ./analysis/asym_multi
bash run_test.sh ./logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/test/avg/ckpt_best.pth
```


## Package Structure

`training/detectors/` contains only `asym_multi_detector.py` and the base detector class; `training/dataset/` contains only the base dataset and image transformations; `training/networks/` contains only the registry interface (the detector loads the local CLIP model for the ViT backbone).
