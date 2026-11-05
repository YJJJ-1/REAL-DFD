# asym_multi

This package follows the DeepfakeBench `training/` directory structure and includes only the entry points, registry, datasets, detectors, trainer, metrics code, and YAML files required for asym_multi. It uses the actual parameters recorded in `/data2/Mining/logs/training/asym_multi_vggface2/training.log`. Compared with the original experiment, the following changes have been made as requested: `use_unpaired_real_for_confidence` is set to `false`, the training batch size is set to 4, the unpaired batch size is set to 16, and the number of test datasets is reduced to 7. These changes will produce new model weights and evaluation metrics. Results obtained using the old weights must not be presented as results from the new experiment.

## External Data and Environment

The package does not include code for other detection methods, datasets, dataset JSON files, auxiliary VGGFace2 images, pretrained CLIP models, old checkpoints, or any result files. It uses resources already available under `/data2/Mining` on the local machine:

- `/data2/Mining/datasets/`
- `/data2/Mining/preprocessing/dataset_json/`
- `/data2/Mining/real-candidates/` (contains preprocessed VGGFace2 images)
- `/data2/Mining/active_learning/clip-model/`

Run the package in an existing DeepfakeBench Python environment. The paths for data and logs in `training/config/train_config.yaml` and `test_config.yaml` have been set to absolute paths. When running on another machine, update these two files as well as the `pretrained` and `unpaired_real_dir` settings in the detector YAML file.

## Extraction and Training

```bash
cd /data2/Mining/analysis
unzip asym_multi.zip
cd asym_multi
bash run_train.sh
```

To specify eight GPUs, use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_train.sh`. The script uses `torchrun --nproc_per_node=8`. The current configuration trains on FF++ c23, uses 8/32 frames for training/testing respectively, a training batch size of 4, an unpaired batch size of 16, a test batch size of 32, and a random seed of 1024.

Logs for a new run are stored in `/data2/Mining/logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/`. Only the training logs, training TensorBoard logs, and checkpoint with the best average AUC are saved.

## Testing with the Best Checkpoint Only

Replace the path below with the directory of the actual new run. This command does not perform training; it only loads `test/avg/ckpt_best.pth` and prints evaluation metrics to the terminal:

```bash
cd /data2/Mining/analysis/asym_multi
bash run_test.sh /data2/Mining/logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/test/avg/ckpt_best.pth
```

Evaluation covers the seven datasets listed in the test YAML file: Celeb-DF-v1, Celeb-DF-v2, DFDC, DFDCP, FaceForensics++, UADFV, and WDF. To evaluate only on FF++, append `FaceForensics++` after the checkpoint path. The `--no-save-results` argument prevents saving prediction scores and summary CSV files, while `--config-loader` uses the batch size of 32 and 0 workers specified in the YAML file. To evaluate the checkpoint from the previous experiment, replace the path with `/data2/Mining/logs/training/asym_multi_vggface2/test/avg/ckpt_best.pth`. However, those results do not represent the newly retrained model.

## Package Structure

`training/detectors/` contains only `asym_multi_detector.py` and the base detector class; `training/dataset/` contains only the base dataset and image transformations; `training/networks/` contains only the registry interface (the detector loads the local CLIP model for the ViT backbone).
