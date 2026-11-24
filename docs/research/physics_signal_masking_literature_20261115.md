# 物理信息构造信号掩码的文献核查

检索日期为2026-11-15。本次只做公开文献检索和方法核对，没有修改实验代码，没有新增训练或目标推理。PU_0、PU_0_concat和PU_1的冻结结果保持不变。

## 筛选标准与主要判断

用户关注的是物理知识参与构造输入掩码，将信号与掩码逐元素相乘后，再由后续网络学习。检索同时使用physics-informed masking、physics-guided spectral mask、fault characteristic frequency weighting、raw signal masking等关键词。

需要分别识别以下情况。

| 类型 | 计算形式 | 是否满足关注点 |
|---|---|---|
| 时域逐点掩码 | x_w(t)=m(t)x(t)，后接网络 | 严格满足时域原始波形加权 |
| 频域掩码后恢复波形 | x_w=IFFT(M·FFT(x))，后接网络 | 满足信号级过滤，但不是时域逐点乘法 |
| 输入变换域加权 | Z_w=M·Z，Z为频谱、包络谱、倒谱或时频图，后接网络 | 满足网络前输入加权，不能称直接加权原始时域波形 |
| 中间层或注意力掩码 | 掩码作用于隐藏特征或注意力矩阵 | 相关方法，不能作为同一种输入前端 |
| 物理标签或随机遮挡 | 掩码生成监督目标或数据增强 | 不自动满足物理掩码参与推理的要求 |

这次最直接的轴承参考为Lu等2023年的PICNN、Ullah等2026年的Conformer-PhyFaultNet，以及Li等2026年的DMC-EN。前两项已核对相关方法正文，DMC-EN仅取得出版商摘要。严格以故障机理确定每个时刻权重的原始时域掩码，本次没有核实到可直接照搬且推理信息条件清楚的轴承分类方法。这个结论是检索范围内的结果，不是证明此类文献不存在。

## 一、与输入物理加权直接相关

### 1. Lu等，2023，PICNN

Hao Lu, Venkat Pavan Nemani, Vahid Barzegar, Cade Allen, Chao Hu, Simon Laflamme, Soumik Sarkar, Andrew T. Zimmerman. A physics-informed feature weighting method for bearing fault diagnostics. Mechanical Systems and Signal Processing, 191, 110171.

[DOI](https://doi.org/10.1016/j.ymssp.2023.110171) · [作者机构全文](https://faculty.sites.iastate.edu/laflamme/files/inline-files/Laflamme_J114_2023.pdf) · [作者代码](https://github.com/SalieriLu/BearingFaultDiagnostics)

这是用户此前提供、PU_1已参考的论文，不是本次新发现。正文第7页、2.2.1节、式5至7明确给出输入加权。故障频率及其谐波约束高斯权重中心，幅值与宽度可训练。输入为包络阶次谱，各加权谱和原谱相加后送入CNN。远离相关频率的权重基线为1，所以是软增强，不是将其余频带全部置零。

可借鉴点是用物理知识限制权重函数的位置和形状。不能把它说成直接对原始振动时序逐点掩码。本文正文式7的相加与作者公开示例的沿谱轴拼接存在实现差异，已在PU_1原文核查中登记，不能不加说明地称原样复现。

### 2. Ullah等，2026，Conformer-PhyFaultNet

Rizwan Ullah, Hazrat Bilal, M. S. Aslam, Sarra Ayouni, Abdul Majid, Athanasios V. Vasilakos, Thippa Reddy Gadekallu. Conformer-PhyFaultNet: Physics-Informed Spectral Attention Conformer for Generalizable Bearing Fault Diagnosis. IEEE Internet of Things Journal, 13(8), 17538–17552.

[DOI](https://doi.org/10.1109/JIOT.2026.3662413) · [作者上传的全文](https://www.researchgate.net/publication/400632677_Conformer-PhyFaultNet_Physics-Informed_Spectral_Attention_Conformer_for_Generalizable_Bearing_Fault_Diagnosis) · [作者机构书目信息](https://pure.ku.edu.kw/en/publications/conformer-phyfaultnet-physics-informed-spectral-attention-conform/)

已取得作者上传的接受稿正文。III-B节、式7和8以理论故障频率及其谐波为中心构造高斯谱掩码，再混合保留原谱。可概括为S_w(f,t)=S(f,t)[αM_phys(f)+(1−α)]，加权时频表示再进入卷积与Conformer。论文使用CWRU、Paderborn和HUST，因而与PU研究直接相关。

需要将输入谱掩码与文中后续的注意力偏置、物理token、训练预训练和预测投票区分。只借鉴前端不等于复现全模型。本次没有取得可核验实现源码；单个f_fault在多类别推理时的完整组合方式，以及带宽、混合系数的实现约束，还需要进一步核对，不能自行补齐后当作作者公开设置。

用户随后追问是否只用振动。再次核对作者接受稿，表I的Paderborn行在Sensors Used列明确列有Accelerometer、Current、Torque，但III-A的预处理以及III-B的输入构造均围绕振动信号描述，未清楚交代电流和转矩的网络输入、各自掩码或融合方式。因此，当前既不能严谨断言所有实验只用了振动，也不能将其登记为已核实的振动与电流多模态融合方法。可能是数据集传感器清单与实际使用通道混写，但这只是解释，不是已确认事实。后续需以源码、最终出版全文的补充说明或作者答复核实。

### 3. Li等，2026，DMC-EN

Longhui Li, Qingbin Tong, Xuedong Jiang, Baohua Wang, Jianjun Xu, Jingyi Huo. Dynamic Mask Cepstrum-Enhancement Network (DMC-EN): A dual-task physics-guided method based on cyclostationary features of bearing signals. Transactions of the Institute of Measurement and Control. First published online 11 January 2026.

[出版商页面](https://journals.sagepub.com/doi/10.1177/01423312251408555)

出版商摘要明确描述，用信号周期性先验结合数据特征和结构评分生成自适应掩码，作用于倒谱后将处理结果送入主干。网络同时做分类与增强时域信号重建，重建包络谱稀疏性约束反向影响掩码学习。

这篇与用户的构想接近，物理规律参与掩码生成，加权发生在主干之前。当前只能确认摘要描述，不能声称已核实掩码公式、带宽范围、三个公开数据集名称、PU划分或开源实现。它也不是直接在原始时域波形上相乘。

## 二、对信号施加掩码，但物理来源或处理对象有所不同

### 4. He等，2024，pyDSN与MDSTFT

Chao He, Hongmei Shi, Ruixin Li, Jianbo Li, ZuJun Yu. Interpretable modulated differentiable STFT and physics-informed balanced spectrum metric for freight train wheelset bearing cross-machine transfer fault diagnosis under speed fluctuations. Advanced Engineering Informatics, 62, 102568.

[DOI](https://doi.org/10.1016/j.aei.2024.102568) · [作者全文](https://liguge.github.io/assets/pdf/advei102568.pdf) · [作者代码](https://github.com/liguge/PyDSN)

已核对第4至6页的方法正文。式12至16通过掩码矩阵控制不同信号片段的有效窗长，再计算STFT，将时频图送入后续迁移诊断网络。物理引导主要来自时频分辨率和谱能量聚集性质，以及BSQ正则化。

它提供可学习信号前端的参考，但没有按BPFO、BPFI位置生成通带掩码。正文还明确不要求转速和轴承几何信息。因此，不能将其归为基于已知故障特征频率的掩码法。它也没有把整个窗口先生成一个独立的加权波形再直接送入1D CNN。

### 5. Zhang等，2026，PhysFormer

Jie Zhang, Yinhui Yu, Sumin Wu. Physics-informed deep learning framework for vibration-based cylinder pressure reconstruction and heat release rate prediction in diesel engines. Frontiers in Mechanical Engineering, 12.

[全文与DOI](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1859799/full)

全文3.1至3.2节先根据振动与缸压的小波相干性保留相关系数较高的系数区域，再重建时域振动，并将处理后振动用作网络输入。文中还另设注意力掩码和热力学损失，两者不能与输入预处理混为一谈。

它是先对变换域信号施加机理相关筛选、再输入网络的跨领域实例，但任务是缸压重建和热释放率预测，不是轴承分类。掩码来自配对测量的相干性，不是仅凭轴承几何公式得出。迁移时必须说明掩码定标需要哪些测量，且只能用训练数据估计。

### 6. Li等，2026，PG-AFNet

Heng Li, Hua Liu, Liang Zhu, Xiangyu Zhao, Lemiao Qiu, Shuyou Zhang. A Physics-Guided Dual-Stream Vibration Feature Fusion Network for Chatter-Induced Surface Mark Diagnosis in Wafer Thinning. Machines, 14(4), 404.

[全文与DOI](https://www.mdpi.com/2075-1702/14/4/404)

全文第3节以加工动力学限定VMD分量选择频带，并将CWT系数限制在物理相关频率范围内，再把处理后的时频图输入二维分支。一维分支输入选取的VMD动态分量。这可以视为固定物理ROI筛选的参考。

该做法更接近硬裁剪和预处理，不是可训练连续掩码。两个分支都来自同一种振动测量，不能当作振动与电流双模态证据。其加工频带不能直接用于PU轴承。

## 三、检索命中但不能混为同类

### 7. Xu等，2024，PIPDN

Zifei Xu, Kaicheng Zhao, Jin Wang, Musa Bashir. Physics-informed probabilistic deep network with interpretable mechanism for trustworthy mechanical fault diagnosis. Advanced Engineering Informatics, 62, 102806.

[DOI](https://doi.org/10.1016/j.aei.2024.102806) · [机构全文](https://researchonline.ljmu.ac.uk/id/eprint/24670/1/Physics-informed%20probabilistic%20deep%20network%20with%20interpretable%20mechanism%20for%20trustworthy%20mechanical%20fault%20diagnosis.pdf)

第5页算法1和2.2节表明，物理频率掩码用于生成训练的物理标签，且标签构造使用已知故障类别。主网络输入仍来自原始信号的FFT实部和虚部。它不是用无标签可得的物理输入掩码再做分类的直接先例。

若把这种由真实类别选择的掩码移到测试输入端，会引入标签泄漏。训练中用标签构造监督目标本身不等于泄漏，问题在于推理时是否使用了待预测类别。

### 8. Mayo等，2026，AF-iTransformer

Sohaib Arshad Mayo等. Let the Model Choose Its Own Frequency: An Adaptive Frequency-Aware Inverted Transformer for Noise-Robust Gearbox Fault Diagnosis. Sensors, 26(14), 4622.

[全文与DOI](https://www.mdpi.com/1424-8220/26/14/4622)

全文3.3节给出FFT、掩码相乘、iFFT和后续Transformer的结构，最适合说明如何保留时域网络接口。但掩码由RMS、峰峰值、谱质心及全局可学习参数生成，不由理论故障频率明确限定位置。物理动机不等于物理公式约束，不能直接作为强物理掩码的证据。

## 四、对当前课题的判断

现有文献已包含根据故障频率构造输入软掩码，再由网络学习的思路。因此，单纯把中间通道调制移到输入端，或把加权函数改名为掩码，不足以主张方法创新。

本次文献最有用的区分是物理位置约束和数据学习的分工。可以用物理知识限定允许增强的频率或周期区域，再学习带宽、强度或原信号保留比例。但这是依据文献提出的设计判断，本次没有实施或证明其在当前PU数据上有效。

如果要求保留1D时域CNN，可使用复频谱掩码后逆变换的形式。它与直接对时域逐点乘法不同。振动的故障重复频率通常在包络分析中解释，原始波形还包含受调制的共振载波。把包络谱中的故障频率位置直接用作原始振动谱的窄通带，并没有被Lu等的论文证明合理。振动和电流应分别明确加权域及物理频率关系。

如果坚持时域周期掩码，仅知道故障频率通常不能确定第一个冲击发生在哪个采样点，还需要相位、起始冲击或可学习对齐。不能用真实测试类别确定周期模板。可操作的无标签推理方式应同时保留各候选故障的模板，或在共享约束下从当前输入估计。

建议精读顺序为Conformer-PhyFaultNet的输入谱掩码、Lu等的受约束加权函数、DMC-EN的周期性和重建反馈。MDSTFT用于补充可学习信号处理前端的背景。不同论文的准确率、数据划分、训练阶段及是否使用目标数据不同，本次不做分数排名。

## 核查范围说明

Conformer-PhyFaultNet使用作者上传接受稿，书目信息用作者机构页面交叉核对。DMC-EN未取得方法全文，只依据出版商摘要。其余列入正文分析的条目核对了公开相关方法段落。网页工具提供PDF文本；截图调用没有返回可供本轮检查的页图，所以本次不声称完成这些PDF的视觉版式核查，也不把排版抽取后的公式符号当作独立数值验证。

部分其他命中只出现在聚合索引、专利或来源不明的页面，未纳入核心推荐。未将随机遮挡自监督、事后可解释掩码或仅隐藏层注意力论文混充输入物理掩码。
