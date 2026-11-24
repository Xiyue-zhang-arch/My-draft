# CWRU实验复现与归档入口

[2026-10-20 的后续设计](FUTURE_PROTOCOL.md)记录当时由 V0 转向全样本基线的安排。已保留的后续版本和执行状态见 [CWRU 实验说明](../../实验说明.md)。


## 实验编号

代码目录均相对本目录的experiments，结果目录均相对项目级experiments/CWRU/实验结果/CWRU。所有638模型及正式成果冻结。

|编号|代码入口|结果与报告|模型数|数据使用说明|
|---|---|---|---:|---|
|实验1|[两套协议入口](experiments/experiment_1/README.md)|[实验1总入口](../../实验结果/CWRU/experiment_1/README.md)|66|[来源与用量](../../实验结果/CWRU/experiment_1/DATA_USAGE.md)|
|实验2|[四方法比较](experiments/experiment_2/matched/README.md)|[报告](../../实验结果/CWRU/experiment_2/matched/REPORT.md)|240|[来源与用量](../../实验结果/CWRU/experiment_2/DATA_USAGE.md)|
|实验3|[随机窗口划分](experiments/experiment_3/README.md)|[原报告](../../实验结果/CWRU/experiment_3/REPORT.md) · [补充](../../实验结果/CWRU/experiment_3_data_small/REPORT.md)|120|[原数据说明](../../实验结果/CWRU/experiment_3/DATA_USAGE.md) · [补充用量](../../实验结果/CWRU/experiment_3_data_small/DATA_USAGE.md)|
|实验4|[按负载排除实验](experiments/experiment_4/README.md)|[报告](../../实验结果/CWRU/experiment_4/REPORT.md)|66|[来源与用量](../../实验结果/CWRU/experiment_4/DATA_USAGE.md)|
|实验5|[随机窗口训练预算匹配](experiments/experiment_5/README.md)|[报告](../../实验结果/CWRU/experiment_5/REPORT.md)|66|[来源与用量](../../实验结果/CWRU/experiment_5/DATA_USAGE.md)|
|实验6|[代码与说明](experiments/experiment_6/README.md)|[报告](../../实验结果/CWRU/experiment_6/REPORT.md)|80|[来源与用量](../../实验结果/CWRU/experiment_6/DATA_USAGE.md)|
|信号分析|[描述性分析](experiments/signal_analysis/README.md)|[报告](../../实验结果/CWRU/signal_analysis/REPORT.md)|0|[来源与用量](../../实验结果/CWRU/signal_analysis/DATA_USAGE.md)|

实验1内部clean_only和awgn_mixed分别保留原33模型及其报告。实验2的提出方法引用实验1评价，不增加模型计数。实验3至5排除OR014_3，实验1保留完整200窗测试。

## 从原始数据训练

使用已有pytorch_GPU环境，从项目根目录运行。以下命令是复现入口，不代表允许自动重训现有结果。

|编号|训练命令|
|---|---|
|实验1，纯干净|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_1/clean_only/train_clean_only.py`|
|实验1，AWGN|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_1/awgn_mixed/train_awgn_mixed.py`|
|实验2|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_2/matched/train_matched_comparison.py`|
|实验3|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_3/run_window_split.py`|
|实验4|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_4/run_exclusion.py`|
|实验5|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_5/run_budget_matched.py`|
|实验6|`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_6/run_transfer.py`|


从只读的 `data/raw/CWRU/48k Drive End Bearking Fault Data` 加载 48 kHz 驱动端数据，每窗 24000 点。默认在 `experiments/CWRU/实验结果/CWRU` 下生成带时间戳的新批次，也可用 `--output` 指定新的输出目录。输出保护覆盖原始数据、实验代码和已有批次。

实验1/2公共入口支持--data-root、--methods、--ratios、--seeds、--subset-seeds、--updates、--device等参数。实验2另支持--protocols。其余入口按自身--help和固定协议使用，不能把不同阶段参数混为一套默认协议。实验5固定600更新、每50更新验证，不提前停止。

## 输入与推理

类别编码Normal=0、IR=1、OR=2、B=3。原按负载读取规则为0/1HP训练、2HP验证、3HP测试，实验3/5按记录内随机重新分配。自定义记录表及所有缓存、划分和派生数据必须保存于data之外，不建立目录联接。

各实验infer入口支持新权重，例如：

```powershell
python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_5/infer.py --model 新模型目录/model.pt --input 待推理.npz --output experiments/CWRU/实验结果/CWRU/新的推理目录
```

NPZ输入包含waveforms[N,1,24000]和rpm[N]，可选labels[N]。输入未逐窗标准化的观测，程序完成归一化。有标签时输出指标和比例混淆矩阵；输出必须为独立新目录。

实验1原冻结模型的model.pt仅含网络权重，物理标准化另存scaler.npz，通过clean_only/infer_clean_only.py或awgn_mixed/infer_awgn_mixed.py的load_pipeline加载。实验2原冻结权重通过matched/infer_comparison.py的load_pipeline加载。实验3至5使用包含method、model_config及完整state_dict的新格式，公共load_model或--model可直接加载。实验4的infer.py按当前归档位置选择权重。

## 公共实现与输出保护

shared包含数据读取、模型、212维物理描述、归一化、抽样、训练、推理和绘图。实验入口只保存代码与必要说明，模型、预测、指标、图表均在项目级experiments/CWRU/实验结果/CWRU。输出保护集中于shared/experiment_paths.py，不修改data，不覆盖正式报告。

当前模型无质量q，完整融合53068参数、数据单分支33964参数。不恢复已删除模型、细分消融或定向频谱修正。两个分支观察同一波形。AWGN每次出现独立50%干净，否则SNR连续U[-4,4]dB，单批一次CE，不使用配对拼接。

训练预算、验证标签预算、选模目标、测试支持数、噪声重复和计算开销按各正式协议披露。新入口在线提取物理特征，旧实验1的干净训练使用缓存且旧轮计时排除预热，不能直接排序。方法比较为适配实现，不称官方精确复现或确认性SOTA。

归档仅变更组织和路径。冻结结果中的历史配置路径不回写，权重、标准化、预测、数值表、正式报告和图表全部原样保留。目录整理不消除历史测试曝光，不改变随机窗口的记录依赖、事后排除等证据边界。

原实验6、实验7及其OR014_1附加排除分析已于2026-10-19按用户要求清除，包含源码、96个模型、预测、指标、图表和报告，不保留备份。删除不消除历史测试曝光。新的单一来源负载迁移支线已完成并冻结，使用全部40记录，共同73训练窗、两协议两模型5种子，不做小样本；同来源模型评价三个目标，共80模型、240评价、12方向，3240逐次指标、168汇总混淆计数及21张总览图表。来源验证19/39/40/40，不声称验证预算跨来源相同。不得自动重训、重新测试或覆盖正式报告及冻结成果。

实验3数据单分支小样本补充入口为`python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_3/supplement_data_small.py`，仅写新独立目录。本轮54模型已完成并冻结，原66模型与预测未改变。新增配对比较沿用原早停规则，不声称实际更新数完全一致，不自动重训或重新测试。
