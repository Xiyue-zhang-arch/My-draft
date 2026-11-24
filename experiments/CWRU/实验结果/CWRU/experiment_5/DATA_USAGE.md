# 实验5 数据集使用说明

本说明依据已有记录、划分及训练子集表整理，不重训、不重新测试、不修改原始数据或已有数值。窗口数均指不同原始窗口，训练种子、模型数和噪声重复不增加独立数据。

原始文件统一位于项目级data/raw/CWRU/48k Drive End Bearking Fault Data。仅使用指定的驱动端DE变量和转速信息，不使用FE信号作为模型输入。48kHz采样，每窗24000点即0.5秒，无重叠。start_sample含起点、end_sample不含终点，不足一窗的末尾不参与分类。

现存66模型，数据依据为[windows.csv](windows.csv)和[records.csv](records.csv)。[逐文件使用表](dataset_file_usage.csv)列出全部40个来源文件、DE变量、采样点数、完整窗数、弃用尾部和各集合用量。被排除或未选中的文件仍保留一行并标0。

## 集合数量

|划分|集合|不同窗口数|不同MAT文件数|
|---|---|---|---|
|standard|训练候选池|467|39|
|standard|验证集|100|39|
|standard|测试集|100|38|
|standard|未用验证候选|0|0|
|standard|最大预算实际训练|287|39|

各类实际窗口数如下。

|划分|集合|正常|内圈|外圈|滚动体|
|---|---|---|---|---|---|
|standard|训练候选池|49|138|133|147|
|standard|验证集|11|30|27|32|
|standard|测试集|10|29|30|31|
|standard|最大预算实际训练|30|77|90|90|

## 实际训练子集

[训练子集逐文件使用表](training_subset_file_usage.csv)逐预算、逐子集种子给出实际使用的MAT文件和每文件窗数。used_train_windows为实际训练用量，0表示该子集未使用；representative_selected_windows指向已有精确窗口表。相同子集跨协议、模型或训练种子复用时只计一份数据。

|划分|预算比例|子集种子|训练窗|MAT文件数|正常/内圈/外圈/滚动体|共享模型数|精确窗口表|
|---|---:|---:|---:|---:|---|---:|---|
|standard|10%|4101|29|29|3/8/9/9|6|[selected_windows](awgn_mixed/fusion/ratio_010_subset_4101_seed_1706/selected_windows.csv)|
|standard|10%|4102|29|29|3/8/9/9|6|[selected_windows](awgn_mixed/fusion/ratio_010_subset_4102_seed_1706/selected_windows.csv)|
|standard|10%|4103|29|29|3/8/9/9|6|[selected_windows](awgn_mixed/fusion/ratio_010_subset_4103_seed_1706/selected_windows.csv)|
|standard|20%|4101|58|39|6/16/18/18|6|[selected_windows](awgn_mixed/fusion/ratio_020_subset_4101_seed_1706/selected_windows.csv)|
|standard|20%|4102|58|39|6/16/18/18|6|[selected_windows](awgn_mixed/fusion/ratio_020_subset_4102_seed_1706/selected_windows.csv)|
|standard|20%|4103|58|39|6/16/18/18|6|[selected_windows](awgn_mixed/fusion/ratio_020_subset_4103_seed_1706/selected_windows.csv)|
|standard|50%|4101|144|39|15/39/45/45|6|[selected_windows](awgn_mixed/fusion/ratio_050_subset_4101_seed_1706/selected_windows.csv)|
|standard|50%|4102|144|39|15/39/45/45|6|[selected_windows](awgn_mixed/fusion/ratio_050_subset_4102_seed_1706/selected_windows.csv)|
|standard|50%|4103|144|39|15/39/45/45|6|[selected_windows](awgn_mixed/fusion/ratio_050_subset_4103_seed_1706/selected_windows.csv)|
|standard|100%|0|287|39|30/77/90/90|12|[selected_windows](awgn_mixed/data_only/ratio_100_subset_0_seed_1706/selected_windows.csv)|

共享模型数不是新增窗口。验证集不随小样本预算缩减，标注预算必须分别列出训练和验证用量。

## 原始文件与各集合对应

以下列出全部40个来源文件。训练候选池与最大实际训练预算分开，不能把候选文件数误报为各小样本的已用文件数；小样本实际来源另见上面的训练子集CSV。

### standard

|MAT文件|DE变量|完整窗|训练候选|最大预算训练|验证|测试|未用验证|训练候选未用|排除|
|---|---|---|---|---|---|---|---|---|---|
|B007_0.mat|X122_DE_time|10|7|7|1|2|0|0|0|
|B007_1.mat|X123_DE_time|20|14|7|3|3|0|7|0|
|B007_2.mat|X124_DE_time|20|14|7|3|3|0|7|0|
|B007_3.mat|X125_DE_time|20|14|8|3|3|0|6|0|
|B014_0.mat|X189_DE_time|10|7|7|2|1|0|0|0|
|B014_1.mat|X190_DE_time|20|14|8|3|3|0|6|0|
|B014_2.mat|X191_DE_time|20|14|8|3|3|0|6|0|
|B014_3.mat|X192_DE_time|20|14|7|3|3|0|7|0|
|B021_0.mat|X226_DE_time|10|7|7|2|1|0|0|0|
|B021_1.mat|X227_DE_time|20|14|8|3|3|0|6|0|
|B021_2.mat|X228_DE_time|20|14|8|3|3|0|6|0|
|B021_3.mat|X229_DE_time|20|14|8|3|3|0|6|0|
|IR007_0.mat|X109_DE_time|10|7|7|2|1|0|0|0|
|IR007_1.mat|X110_DE_time|20|14|7|3|3|0|7|0|
|IR007_2.mat|X111_DE_time|20|14|7|3|3|0|7|0|
|IR007_3.mat|X112_DE_time|20|14|7|3|3|0|7|0|
|IR014_0.mat|X173_DE_time|2|1|1|1|0|0|0|0|
|IR014_1.mat|X175_DE_time|15|11|6|2|2|0|5|0|
|IR014_2.mat|X176_DE_time|20|14|7|3|3|0|7|0|
|IR014_3.mat|X177_DE_time|20|14|7|3|3|0|7|0|
|IR021_0.mat|X213_DE_time|10|7|7|1|2|0|0|0|
|IR021_1.mat|X214_DE_time|20|14|7|3|3|0|7|0|
|IR021_2.mat|X215_DE_time|20|14|7|3|3|0|7|0|
|IR021_3.mat|X217_DE_time|20|14|7|3|3|0|7|0|
|Normal_0.mat|X097_DE_time|10|7|7|2|1|0|0|0|
|Normal_1.mat|X098_DE_time|20|14|8|3|3|0|6|0|
|Normal_2.mat|X099_DE_time|20|14|8|3|3|0|6|0|
|Normal_3.mat|X100_DE_time|20|14|7|3|3|0|7|0|
|OR007_0.mat|X135_DE_time|10|7|7|1|2|0|0|0|
|OR007_1.mat|X136_DE_time|20|14|8|3|3|0|6|0|
|OR007_2.mat|X137_DE_time|20|14|8|3|3|0|6|0|
|OR007_3.mat|X138_DE_time|20|14|9|3|3|0|5|0|
|OR014_0.mat|X201_DE_time|10|7|7|1|2|0|0|0|
|OR014_1.mat|X202_DE_time|20|14|9|3|3|0|5|0|
|OR014_2.mat|X203_DE_time|20|14|9|3|3|0|5|0|
|OR014_3.mat|X204_DE_time|20|0|0|0|0|0|0|20|
|OR021_0.mat|X238_DE_time|10|7|7|1|2|0|0|0|
|OR021_1.mat|X239_DE_time|20|14|9|3|3|0|5|0|
|OR021_2.mat|X240_DE_time|20|14|8|3|3|0|6|0|
|OR021_3.mat|X241_DE_time|20|14|9|3|3|0|5|0|

实际最大预算训练与验证、训练与测试、验证与测试的原文件交集数分别为39、38、38。

## 纳入及解释边界

整条排除OR014_3.mat，其20窗不进入训练、验证或测试。原始MAT仍只读保留。事后排除及已有测试曝光不会因重划分或重训消失。
记录内先切窗再70%/15%/15%划分，同一MAT可贡献多个集合。无采样点重叠不等于记录独立，不能据此声称跨记录或跨负载泛化。IR014_0.mat只有2个完整窗，分为1训练、1验证、0测试。
467窗为候选池，最大实际训练287窗，余180候选窗未被该最大预算模型使用。小样本29/58/144窗不是467的10%/20%/50%。同一小样本种子的三预算嵌套；满预算选样种子0不同，不能默认全部小子集都包含在287窗满预算内。
小样本从头训练，只拟合当前子集的标准化参数。实验1、3、4、5的数据单分支仅最大预算，不能据此证明物理分支提高小样本效率；实验2四比较方法按保存预算共享对应训练子集。
人工AWGN不增加原始文件或独立窗口。纯干净协议不人工加噪测试；AWGN协议测试干净及−4/−2/0/2/4dB，每噪声档5实例。训练有放回抽样的观测次数不同于不同训练窗口数。

原始记录不等于已经验证的独立轴承身份，窗口重复、模型重复和不同负载记录不能直接视为独立轴承重复。
