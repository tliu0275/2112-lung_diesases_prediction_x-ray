# 2112-lung_diesases_prediction_x-ray

## 如何用你本地 `pneumonia/` 文件夹训练

你的数据结构是：

```
pneumonia/
  Train_viruspneumonia_2112/
    *.jpg / *.png / *.dcm ...
  Train_bacteriapneumonia_2112/
    *.jpg / *.png / *.dcm ...
```

这属于**两分类**（病毒性 vs 细菌性）。本项目已改为支持“**直接读取子文件夹作为类别**”，不需要 CSV。

### Step 0：把数据放到项目旁边（推荐）

把 `pneumonia/` 放在项目根目录下（和 `src/`、`configs/` 同级）。

如果你不想放在项目里，也可以放任意位置，后面在 `configs/config.yaml` 里改 `data.data_dir` 为绝对路径即可。

### Step 1：创建环境并安装依赖

在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

（macOS + Apple Silicon 如果安装某些包报错，建议先升级 pip：`pip install -U pip` 再装依赖。）

### Step 2：配置数据路径

打开 `configs/config.yaml`，找到：

- `data.data_dir`: 改成你的 `pneumonia` 路径（相对或绝对都行）
- `data.csv_file`: 保持 `null`（表示走“文件夹扫描”模式）

默认配置已经是：

- `model.model_type: binary`
- `model.name: densenet121`

### Step 3：开始训练

在项目根目录执行：

```bash
python -m src.training.train --config configs/config.yaml --device cpu
```

如果你有 NVIDIA GPU（CUDA 可用）：

```bash
python -m src.training.train --config configs/config.yaml --device cuda
```

训练输出：

- 最佳模型会保存到 `checkpoints/best_model.pth`
- 日志会输出到控制台（后续可扩展到 TensorBoard）

### Step 4：常见问题

- **我只有两个文件夹，没有 Normal 类**：没问题，本项目会把两个子文件夹当成两类来训练（病毒 vs 细菌）。
- **我有更多类别**：也可以，只要 `data_dir/` 下有多个子文件夹，都会被当作类别（按文件夹名排序映射到 0..N-1）。
- **图片不止 jpg/png**：支持 `jpg/jpeg/png/dcm/bmp/tif/tiff/webp`。
