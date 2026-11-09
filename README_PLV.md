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


全频段采用 0.5–45 Hz 的宽带 Hilbert 相位 PLV。宽带含多种振荡，瞬时相位的生理解释弱于窄带 PLV，单独报告，不与五子频段混成同一种解释。此前短训练的缓存和检查点已清理；六频段必须使用独立数据目录和运行目录，例如 runs/tcn_6bands。



## 代码布局

- main_plv.py：训练入口；inference_plv.py：检查点评测入口。
- engine_plv.py：训练、验证、基线选择和结果导出。
- models/plv_model.py：MLP/TCN。
- dataset_maker/plv_dataset.py：同步、滤波、PLV 标签及数据清单。
- build_plv_data.py：数据构建入口；plv_metrics.py：统计指标。

原 UnEBOLT main.py/inference.py 保留作为 EEG→BOLD 参考。当前实现仅支持跨被试；所有 raw data、缓存、权重和个人 IDE 配置均不提交。正式训练前仍需验证伪迹质量和边缘缓冲。此前短训练结果已清理，不能作为实验结论。


## 实验版本与 HTML 台账

`--output` 始终指向实验根目录，例如 `runs/20261109_tcn_s2026_v1`，训练必须使用空目录。可用 `--notes "比较 TCN 与 MLP"` 记录实验目的。每次实验固定一个模型、数据版本和 seed。

```text
runs/
  index.html                         # 自动生成的跨实验可视化台账
  20261109_tcn_s2026_v1/
    training/
      run.json                       # 参数、时间、完成状态、Git、依赖版本
      manifest.json                  # 训练时数据清单与划分快照
      state.npz                      # 训练集 scaler 和被试等权模板
      best.pt                        # 神经网络最佳权重；ridge 为 ridge.joblib
      training.csv                   # 神经网络每 epoch 日志
      source/                        # 本次执行的 PLV 源文件快照
      source.patch                   # 有已跟踪文件修改时保存 Git patch
    evaluations/
      <UTC时间戳_UUID>_val/
        evaluation.json              # 评估版本、数据校验、检查点哈希、时间与状态
        source/                      # 评测时源文件快照
        val_<scan>_predictions.npz
        val_scans.csv
        val_subjects.csv
        val_summary.csv
      <另一时间戳_UUID>_test/
        ...
```

训练结束后默认评测一次 val。重复调用 inference_plv.py 仅加载 training/ 的产物，每次新建 evaluations/ 子目录，不覆盖训练记录和此前评测。也兼容读取旧版直接位于实验根目录的 state.npz/run.json，但旧实验未记录的数据版本无法追补。

除处理配置签名外，每次训练/评测还对完整 manifest 和清单中的 NPZ 内容计算 SHA-256；若数据内容或划分发生变化，拒绝复用新格式实验的检查点。首次读取大数据会增加校验时间。源码快照保证工作区尚未提交的 PLV 修改也可追溯。数据构建版本仍需结合数据清单和保存的数据文件维护。

训练和评测结束自动重建同级 runs/index.html。也可手动生成或指定输出：

```bash
python experiment_ledger.py --runs runs
python experiment_ledger.py --runs runs --output reports/experiments.html
```

用浏览器打开 HTML 即可离线查看，无需服务器或第三方服务。支持实验名/模型/备注筛选、val/test 切换、频段与指标选择、按指标自动排序、横向条形图、被试 CI 与有效比例、数据/代码版本、完整配置展开及当前表格 CSV 导出。改动实验后需重新生成并刷新浏览器。不同数据清单会提示；即使校验一致，也需人工确认实验目的与统计方案可比较。相同 seed 的重复评测不会当作独立随机种子合并；每份评测单独展示。

运行状态在启动时写入 running、成功后改为 completed；异常或中断留下 running，不能当成已完成实验。台账扫描 evaluations/ 中的评测记录，未完成记录不展示有效分数；只有训练尚未开始评测的目录可直接查看 training/run.json。当前仍仅保存最佳模型参数，不支持恢复 optimizer 状态续训。权重、预测和本地台账位于 Git 忽略的 runs/，需另行备份。
