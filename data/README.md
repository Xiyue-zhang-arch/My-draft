# 数据目录

原始数据包括 CWRU、MAFAULDA、Ottawa、PU 和 only vibration 五套。用于正式实验的数据划分和样本数见 [CWRU 实验说明](../experiments/CWRU/实验说明.md)和 [PU 实验说明](../experiments/PU/实验说明.md)。数据清点按原始文件统计，实验样本数按各协议的实际窗口统计。

|目录|用途|
|---|---|
|raw/CWRU|两档采样率的轴承振动MAT；正常采样率及部分变量混存需先处理|
|raw/MAFAULDA|MAFAULDA原始CSV，不修改|
|raw/Ottawa|Ottawa原始Excel，不修改|
|raw/PU|32个轴承RAR包及说明文档；内部2560个MAT，基础清点已列出成员并抽查32条|
|pu_shared/prepared_inputs|PU 共用的角域输入缓存及样本身份，用于保留的四工况划分和直接拼接实验|
|raw/only vibration|45个四通道振动MAT及说明论文；部分实际时长与论文不同|


