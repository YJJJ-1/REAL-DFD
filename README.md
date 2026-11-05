# asym_multi

This package follows the DeepfakeBench `training/` directory structure and includes only the entry points, registry, dataset, detector, trainer, metrics code, and YAML files required for asym_multi.

## External Data and Environment

The package does not include code for other detection methods, datasets, dataset JSON files, auxiliary VGGFace2 images, or pretrained CLIP models. 

- `./datasets/`
- `./preprocessing/dataset_json/`
- `./real-candidates/` (contains preprocessed VGGFace2 images)
- `./active_learning/clip-model/`


## Extraction and Training

```bash
unzip asym_multi.zip
cd asym_multi
bash run_train.sh
```

To specify eight GPUs, use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_train.sh`. The script uses `torchrun --nproc_per_node=8`. The current configuration trains on FF++ c23 and uses 8/32 frames for training/testing respectively.

You can also call the included DeepfakeBench `training/train.py` directly from the extracted `asym_multi` directory:

```bash
cd ./asym_multi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 training/train.py \
  --ddp \
  --detector_path training/config/detector/asym_multi.yaml \
  --no-save_feat
```

## Testing with the Best Checkpoint Only

Replace `YYYY-MM-DD-HH-MM-SS` with the actual training run timestamp. To test with the saved best checkpoint, run `run_test.sh` from the extracted package:

```bash
cd ./asym_multi
bash run_test.sh ./logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/test/avg/ckpt_best.pth
```

You can also run the included DeepfakeBench `training/test.py` directly. Both commands load only `test/avg/ckpt_best.pth`, do not train, and print metrics without saving result CSV files:

```bash
cd ./asym_multi
CUDA_VISIBLE_DEVICES=0 python training/test.py \
  --detector_path training/config/detector/asym_multi.yaml \
  --weights_path ./logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/test/avg/ckpt_best.pth \
  --config-loader \
  --no-save-results
```
