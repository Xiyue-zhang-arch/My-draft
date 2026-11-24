# 第四个比较方法替换与实施

2026-10-11，用户确认没有WD-KANTF全文或作者代码，并明确同意替换为已提供全文的注意力加权混合特征模型。

来源为An attention-weighted hybrid feature learning framework for bearing fault diagnosis and maintenance cost optimization，Scientific Reports16:27148，doi:10.1038/s41598-026-55303-4。本轮模型名AW_Hybrid_adapted，不能称作者官方实现或声称已复现其原论文分数。

完整保留论文Eq11-37及Table2的诊断结构。db4三级小波包，逐层对称延拓，末端节点按频率排序，8维能量占比加峰度、偏度、峰值因子，共11维。Linear11至64，不加非线性。深度分支使用步长卷积、BN、ReLU，不用池化，之后2层双向GRU，每方向32。物理向量生成单个Query，深度序列生成Key/Value，4头各16维，拼接各头输出，不额外加入论文公式未给出的输出投影。标量门控只接收物理表示，融合alpha*physical+(1-alpha)*attention，分类64至32至4。

卷积的具体层数、通道、核宽与步长未在论文Table2公开。本适配预先固定三层32/64/64通道，核31/9/5，步长8/4/4，以24000点输入产生188步序列，控制长序列GRU成本。这些不是原文明确给出的参数。PyTorch默认初始化与BN默认参数。未引入额外物理统计标准化，公式本身只对波形及能量归一化。

论文的维护成本映射属于诊断输出后的评价，不改变训练损失。本次统一四类分类任务不加入维护成本指标，不改变任一方法的分类阈值。原文示意去噪图不作为未经定义的额外滤波模块实现。

当前数据划分、训练子集、非配对AWGN、验证和测试条件、共同优化预算及复杂度口径见[matched/PROTOCOL.md](matched/PROTOCOL.md)。本模型每协议30个，和其他三个比较方法合计240个冻结模型。现存训练入口为[matched/train_matched_comparison.py](matched/train_matched_comparison.py)，不再保留历史run.py或run_four.py入口。
