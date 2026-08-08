# EMCAD 的 ACDC 隔离适配

这套文件不修改以下 Synapse 文件：

- `train_synapse.py`
- `test_synapse.py`
- `trainer.py`
- `utils/utils.py`
- `utils/dataset_synapse.py`

ACDC 使用自己独立的 `utils/acdc_utils.py`，因此不会改变 Synapse 的损失、验证、测试或指标。

## 一、复制文件

在服务器项目根目录中，先备份当前被多轮粘贴修改过的 ACDC 文件：

```bash
cp train_ACDC.py train_ACDC.py.bak
cp test_ACDC.py test_ACDC.py.bak
cp utils/dataset_ACDC.py utils/dataset_ACDC.py.bak
cp start_train_acdc.sh start_train_acdc.sh.bak
cp start_test_acdc.sh start_test_acdc.sh.bak
```

将 `ACDC_READY` 中的文件复制到下列目标：

```text
ACDC_READY/check_acdc_data.py          -> check_acdc_data.py（新增）
ACDC_READY/train_ACDC.py              -> train_ACDC.py
ACDC_READY/test_ACDC.py               -> test_ACDC.py
ACDC_READY/utils/dataset_ACDC.py      -> utils/dataset_ACDC.py
ACDC_READY/utils/acdc_utils.py        -> utils/acdc_utils.py（新增）
ACDC_READY/start_train_acdc.sh        -> start_train_acdc.sh
ACDC_READY/start_test_acdc.sh         -> start_test_acdc.sh
ACDC_READY/stop_train_acdc.sh         -> stop_train_acdc.sh（新增）
ACDC_READY/stop_test_acdc.sh          -> stop_test_acdc.sh（新增）
```

不要把 `acdc_utils.py` 粘贴进 `utils/utils.py`。

## 二、确认数据布局

必须是：

```text
data/ACDC/
├── train/
├── valid/
├── test/
└── lists/lists_ACDC/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

训练、验证切片以及测试 volume 的 NPZ 都必须包含 `img` 和 `label` 两个键，标签范围必须为 `0..3`。

## 三、先做静态检查

```bash
python -m py_compile \
  check_acdc_data.py \
  train_ACDC.py \
  test_ACDC.py \
  utils/dataset_ACDC.py \
  utils/acdc_utils.py

bash -n start_train_acdc.sh
bash -n start_test_acdc.sh
bash -n stop_train_acdc.sh
bash -n stop_test_acdc.sh
```

再确认参数入口能加载：

```bash
python train_ACDC.py --help
python test_ACDC.py --help
```

然后先检查全部 ACDC 列表、文件维度、NPZ 键和标签范围：

```bash
python check_acdc_data.py \
  --root_path ./data/ACDC \
  --list_dir ./data/ACDC/lists/lists_ACDC
```

只有输出第一行为 `ACDC_DATA_OK` 时才开始训练。

## 四、前台烟雾测试

第一次不要直接后台训练。先执行一个 batch、一个验证 volume：

```bash
python train_ACDC.py \
  --root_path ./data/ACDC \
  --list_dir ./data/ACDC/lists/lists_ACDC \
  --output_dir ./model_pth/ACDC \
  --run_name smoke_acdc \
  --img_size 224 \
  --batch_size 1 \
  --max_epochs 1 \
  --max_train_batches 1 \
  --max_valid_volumes 1 \
  --num_workers 0 \
  --supervision last_layer \
  --no_pretrain \
  --device auto
```

成功标准：

```text
model_pth/ACDC/smoke_acdc/
├── best.pth
├── last.pth
├── epoch_1.pth
├── config.json
├── train.log
├── validation_metrics.csv
└── tensorboard/
```

用烟雾测试 checkpoint 测一个病例：

```bash
python test_ACDC.py \
  --checkpoint ./model_pth/ACDC/smoke_acdc/best.pth \
  --root_path ./data/ACDC \
  --list_dir ./data/ACDC/lists/lists_ACDC \
  --output_dir ./model_pth/ACDC/smoke_acdc/predictions \
  --output_csv ./model_pth/ACDC/smoke_acdc/test_metrics.csv \
  --img_size 224 \
  --inference_batch_size 1 \
  --num_workers 0 \
  --max_cases 1 \
  --save_nii \
  --save_npz
```

## 五、正式训练和测试

烟雾测试成功后启动正式训练：

```bash
CONDA_ENV_NAME=sld_emcad_251 \
CUDA_DEVICE=0 \
BATCH_SIZE=6 \
MAX_EPOCHS=150 \
SUPERVISION=mutation \
bash start_train_acdc.sh
```

训练结束后，从训练日志最后一行取得 `BEST_CHECKPOINT`，再执行：

```bash
CKPT=/绝对路径/model_pth/ACDC/某次运行/best.pth \
CUDA_DEVICE=0 \
bash start_test_acdc.sh
```

测试输出包括：

- 每个病例、每个类别的 Dice、HD95、Jaccard、ASD；
- 汇总均值 `test_metrics.csv`；
- `test.log`；
- 每个病例的预测 NPZ；
- 图像、预测和标签三个 NIfTI 文件。

## 六、当前共享 utils.py 的处理

这套 ACDC 代码完全不导入 `utils/utils.py`。你之前按聊天内容加入的 `metric_fn` 不是本方案所需，可恢复原样；即使暂时保留，Synapse 未传 `metric_fn` 时仍走原默认函数，但为了保持代码与原 Synapse 版本一致，建议删掉那次新增的参数和分支。
