# BOLD → EEG PLV 跨被试 baseline

此实现独立于原 UnEBOLT 入口。保留原 common 参考；输入 MRI 第 k…k+9 帧，目标 EEG 为第 k 到 k+4 个 R149 事件之间的 8.4 秒窗口，步长 1TR。26 通道、六频段（五个子频段加 broadband 0.5–45 Hz）、每频段 325 条严格上三角边。

## WSL 环境

环境：`/home/pc/.venvs/bold-plv`，由 uv 和系统 Python 3.12 创建。Python 3.11 下载失败后采用系统版本。激活：

```bash
source /home/pc/.venvs/bold-plv/bin/activate
cd '/mnt/i/eeg2fmri work/BOLD_to_EEG_PLV'
```

VS Code 建议通过 WSL 打开目录，选择上述环境中的 `bin/python` 作为解释器。环境位于 Linux 文件系统，源码、数据和输出位于 I: 挂载目录。

## 构建数据

```bash
python build_plv_data.py \
  --data-root '/mnt/i/eeg2fmri work/UnEBOLT_code/data/dataset_public_rest' \
  --output artifacts/data_common_10bold_4tr_6bands \
  --reference existing --split scan_split.yaml
```

先对连续信号做四阶 Butterworth SOS 零相位带通，然后反射填充后 Hilbert 提取相位。标签不做逐窗口基线校正，不改参考。默认剔除距离记录两端不足 8.4 秒的标签；该缓冲长度是可配置的工程起点，需通过敏感性检查验证。发现内部 discontinuity 会停止，避免跨断点滤波。

每扫描保存 NPZ：x、y、原始 BOLD frame_index、EEG 起止秒数、配置签名；manifest 保存扫描/被试划分、参数和幅值峰值。跨被试按 YAML 检查被试互斥。BOLD 不做跨时间滤波，标准化在训练入口仅用训练帧估计。没有预设幅值阈值自动剔除伪迹；结果是初步实验，不代表 EEG 已通过完整伪迹质量验收。

`--only-scans sub01-scan01` 可调试单扫描。完整构建时去掉该参数。同目录仅允许相同配置，修改参考/滤波缓冲/窗口等需新输出目录。

## 训练与验证

```bash
python main_plv.py --data artifacts/data_common_10bold_4tr_6bands --output runs/template_6bands --model template --device cpu
python main_plv.py --data artifacts/data_common_10bold_4tr_6bands --output runs/ridge_6bands --model ridge --device cpu
python main_plv.py --data artifacts/data_common_10bold_4tr_6bands --output runs/tcn_6bands --model tcn --epochs 50 --device cuda
```

MLP 使用 `--model mlp`。神经网络 MSE、AdamW、验证集选择最佳 epoch；岭回归仅在验证集搜索 alpha。默认只输出验证结果。均值模板对训练扫描先平均，再按训练被试等权平均。

冻结方案后的测试：

```bash
python inference_plv.py --data artifacts/data_common_10bold_4tr_6bands --output runs/tcn_6bands --model tcn --eval-split test --device cuda
```

输出包含最佳权重、训练日志、train-only scaler/template、配置、逐扫描预测、逐扫描/被试指标和被试 bootstrap CI。时间/空间相关采用 Fisher-z 汇总；零方差相关保留 NaN 并报告有效比例。对比模型时需同时检查有效比例，避免排除失败预测造成偏差。训练选择目前使用验证窗口平均 MSE，报告则按扫描→被试等权汇总。

## 已实现与待补

已实现跨被试数据构建、模板/岭回归/MLP/TCN、主要评测公式和检查。未实现被试内划分、自动伪迹判定、窗口/对齐参数搜索、时间错位对照及模型间统计检验。新建同一任务的另一随机种子运行目录即可避免覆盖实验。

```bash
python -m pytest tests/test_plv_core.py -q
```

已验证环境：Python 3.12.3、PyTorch 2.15.0+cu132；GPU RTX 4070 Ti SUPER 16GB。精确依赖见 requirements-plv.lock.txt。后续安装可用 uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple；当前环境已安装完成。原始 common 参考由用户确认。


全频段采用 0.5–45 Hz 的宽带 Hilbert 相位 PLV。宽带含多种振荡，瞬时相位的生理解释弱于窄带 PLV，单独报告，不与五子频段混成同一种解释。旧五频段数据和检查点保留；六频段必须使用新数据目录和新运行目录，例如 runs/tcn_6bands。



## 代码布局

- main_plv.py：训练入口；inference_plv.py：检查点评测入口。
- engine_plv.py：训练、验证、基线选择和结果导出。
- models/plv_model.py：MLP/TCN。
- dataset_maker/plv_dataset.py：同步、滤波、PLV 标签及数据清单。
- build_plv_data.py：数据构建入口；plv_metrics.py：统计指标。

原 UnEBOLT main.py/inference.py 保留作为 EEG→BOLD 参考。当前实现仅支持跨被试；所有 raw data、缓存、权重和个人 IDE 配置均不提交。正式训练前仍需验证伪迹质量和边缘缓冲。此前短训练结果已清理，不能作为实验结论。
