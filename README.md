# asym_multi VGGFace2 复现包

本包按 DeepfakeBench 的 `training/` 结构仅保留 asym_multi 所需的入口、registry、数据集、检测器、训练器、指标代码及 YAML，使用 `/data2/Mining/logs/training/asym_multi_vggface2/training.log` 的实际参数。相对原实验，按要求将 `use_unpaired_real_for_confidence` 改为 `false`、训练 batch size 改为 4、unpaired batch size 改为 16、测试集缩减为 7 个。这会得到新的权重和指标，不能把旧权重的测试结果当成新实验结果。

## 外部数据与环境

包内不包含其他检测方法的代码、数据集、数据集 JSON、VGGFace2 辅助图像、CLIP 预训练模型、旧 checkpoint 或任何结果文件。使用本机 `/data2/Mining` 已有资源：

- `/data2/Mining/datasets/`
- `/data2/Mining/preprocessing/dataset_json/`
- `/data2/Mining/real-candidates/`（其中的 VGGFace2 预处理图像）
- `/data2/Mining/active_learning/clip-model/`

请在已有的 DeepfakeBench Python 环境中运行。包内 `training/config/train_config.yaml` 和 `test_config.yaml` 已将数据及日志目录改为绝对路径；换机器时需调整这两个文件和 detector YAML 中的 `pretrained`、`unpaired_real_dir`。

## 解压与训练

```bash
cd /data2/Mining/analysis
unzip asym_multi_vggface2_no_conf_repro.zip
cd asym_multi_vggface2_no_conf_repro
bash run_train.sh
```

可用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_train.sh` 指定 8 张卡。脚本使用 `torchrun --nproc_per_node=8`。当前配置为 FF++ c23 训练、训练/测试各 8 帧、训练 batch size 4、unpaired batch size 16、测试 batch size 32、种子 1024。每卡 4 张 paired 图像平均约含 2 张 paired real，因此 unpaired real 与 paired real 的期望比例为 16:2 = 8:1；实际每批会波动。`nEpochs: 8` 在当前代码中执行 epoch 0 到 8，共 9 轮。每轮在配置中的 7 个数据集上验证，并按平均帧级 AUC 更新唯一的 `test/avg/ckpt_best.pth`。训练样本数 46032，8 卡每卡每轮 1439 步；`steps_per_epoch` 已由 959 改为 1439，以保持预热和动态 margin 的 epoch 时点。

新运行日志位于 `/data2/Mining/logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/`。仅保存训练日志、训练 TensorBoard 和平均 AUC 最佳权重；不保存训练/测试 data_dict、逐数据集指标 pickle、逐数据集权重、测试 TensorBoard、特征文件或分数 CSV。

## 仅用最佳权重测试

将下面路径换成实际新运行目录。该命令不训练，只加载 `test/avg/ckpt_best.pth`，并且仅在终端打印指标：

```bash
cd /data2/Mining/analysis/asym_multi_vggface2_no_conf_repro
bash run_test.sh /data2/Mining/logs/training/asym_multi_YYYY-MM-DD-HH-MM-SS/test/avg/ckpt_best.pth
```

测试 YAML 列出的 Celeb-DF-v1、Celeb-DF-v2、DFDC、DFDCP、FaceForensics++、UADFV、WDF 这 7 个数据集。只测 FF++ 可在权重路径后加 `FaceForensics++`。测试参数 `--no-save-results` 不写分数和汇总 CSV，`--config-loader` 使用 YAML 中的 batch size 32、workers 0。若想测试旧实验权重，可把路径换成 `/data2/Mining/logs/training/asym_multi_vggface2/test/avg/ckpt_best.pth`，但它不代表本次重新训练的结果。

本配置沿用原实验的 15 数据集平均 AUC 选权重协议；其中包含跨数据集测试集。严格的泛化实验应另划验证集重新训练。当前环境无 CUDA GPU，包只做了语法、YAML、资源路径和 ZIP 完整性检查，未实际运行训练或推理。

## 包内结构

`training/detectors/` 只有 `asym_multi_detector.py` 和基类；`training/dataset/` 只有基础数据集及图像变换；`training/networks/` 只有 registry 接口（ViT 由检测器调用本地 CLIP）。
