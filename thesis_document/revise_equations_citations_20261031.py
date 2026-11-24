from pathlib import Path
import re, shutil

root=Path(r'D:\Project\MY draft')
src=root/'thesis_document'/'成果'/'V3_毕业论文_噪声补充_20261031'
out=root/'thesis_document'/'成果'/'V3_论文_公式编号与物理依据_20261031'
out.mkdir(parents=True,exist_ok=True)
shutil.copytree(src/'media',out/'media',dirs_exist_ok=True)
s=(src/'thesis.tex').read_text(encoding='utf-8')
def change(old,new):
    global s
    count=s.count(old)
    if count != 1: raise ValueError(f'expected one occurrence, got {count}: {old[:100]}')
    s=s.replace(old,new,1)

change(r'\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc}',r'\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc}'+'\n'+r'\numberwithin{equation}{chapter}'+'\n'+r'\renewcommand{\theequation}{\thechapter-\arabic{equation}}')
change('则常用候选频率可以表示为：','则常用候选频率可由轴承运动学计算，见式（\\ref{eq:auto01}）至式（\\ref{eq:auto04}）{[}1,2{]}：')
change('对转速r，以每分钟转数表示时先计算$f_{r}$=r/60，再以阶次与$f_{r}$相乘得到该窗口的候选频率。','对以每分钟转数表示的转速，将其除以60得到轴转频，再以阶次与轴转频相乘得到该窗口的候选频率。')
change('因此，描述向量维数为4×(6×8+5)=212。','四个频带各产生53维描述，因此描述向量共有212维。')
change('考虑频率的1.5\\%作为相对容差','考虑频率的1.5\\%作为相对容差') if False else None
change('候选谐波h$f_{0}$附近的目标区域宽度不小于2 Hz，同时考虑频率的1.5\\%作为相对容差。','候选谐波h$f_{0}$附近的目标搜索半宽取2 Hz与该谐波频率1.5\\%中的较大值。')
change('该过程不是从测试类别反推故障频率，也没有对外圈和滚动体类别应用专门的测试时修正。','阶次用于跟随转速确定候选频率，与固定赫兹值直接比较不同；利用转速与轴承几何确定缺陷频率的依据见文献{[}1,2{]}。这里使用的六个阶次、具体倍率以及转速缺失时采用标称值属于本文实现的设定，并非文献对所有轴承的通用规定。该过程不是从测试类别反推故障频率，也没有对外圈和滚动体类别应用专门的测试时修正。')
change('为了减少结果对某一预设频带的依赖，本文保留','Duan等指出，故障相关冲击可分布于多个频带，单一频带的选取可能遗漏信息{[}20{]}。据此，为了减少结果对某一预设频带的依赖，本文保留')
change('有效时长限制了谱线分辨率，邻近候选频率的能量可能发生重叠。','上述四个频带边界、滤波器阶数、重采样率、截边长度和窗函数是本实验固定的工程实现，并非文献{[}20{]}给出的统一最优参数；文献采用的是频带选择和源分离，本文没有声称复现该算法。有效时长限制了谱线分辨率，邻近候选频率的能量可能发生重叠。')
change('这种构造有两个目的。','上述组合借鉴了频谱谐波和包络分析{[}1,20{]}、自相关重复结构{[}21,26{]}、事件相位向量{[}22{]}、波形统计{[}1{]}与谱平坦度{[}23{]}等不同来源的基本思想。各文献没有共同提出本文的212维向量；八项周期描述、五项形状描述以及跨频带的组合方式由本文根据现有实现组织。\n\n这种构造有两个目的。')
change('对于候选频率$f_{0}$，代码在一至六倍频附近搜索包络功率谱的局部峰值。','包络谱中的特征频率及其谐波可用于识别重复冲击，文献{[}1,20{]}给出了这一诊断思路。对于候选频率$f_{0}$，本文代码在一至六倍频附近搜索包络功率谱的局部峰值。')
change('相反，在低信噪比或局部频率漂移下，真实冲击也可能只产生少数清晰谐波，因此覆盖率低不能直接判定无故障。','相反，在低信噪比或局部频率漂移下，真实冲击也可能只产生少数清晰谐波，因此覆盖率低不能直接判定无故障。本文的六阶谐波均值、三倍阈值和覆盖率是对谐波证据的具体汇总方式；文献{[}1,20{]}支持考察谐波，但不直接证明这些具体阈值最优。')
change('较高的对比度表示在所设参照下目标位置更突出；该值仍依赖候选阶次和偏移参照的选择，不是独立的显著性检验结果。','利用谱峰相对于局部背景的突出程度是对频谱识别思路的工程化表达{[}1,20{]}。本文进一步构造偏移频率参照并取最大参照值；这些偏移倍率是本文设置，并非上述文献给出的标准。较高的对比度表示在所设参照下目标位置更突出；该值仍依赖候选阶次和偏移参照的选择，不是独立的显著性检验结果。')
change('包络的重复结构还可以通过自相关观察。','包络的重复结构还可以通过自相关观察。Moshrefzadeh和Fasana利用包络平方的自相关研究周期冲击与解调频带选择{[}21{]}；该文支持使用自相关考察重复结构，并不等同于本文对普通去均值包络实施的归一化自相关。')
change('该描述关注时域重复，而非仅依赖包络谱的局部峰。','该描述关注时域重复，而非仅依赖包络谱的局部峰。三个延迟及其局部最大值平均、偏移频率差值是本文的具体构造，不应归为文献{[}21{]}的原始定义。')
change('其中稳健离差通过1.4826倍的绝对中位差近似。','其中稳健离差通过1.4826倍的绝对中位差构造。该比例常用于使MAD在正态分布下成为标准差的一致尺度估计{[}25{]}，但这不保证实际包络服从正态分布。三倍阈值、七点局部峰判定及边界处理属于本文实现。')
change('相位集中度由相位向量平均的模长表示：','将事件映射到候选周期的相位后，向量合成长度可以描述相位集中程度；Leahy等对事件相位向量及周期检验给出了相关方法依据{[}22{]}。本文采用相位向量平均模长作为输入特征，而没有沿用该文的显著性检验或其概率结论：')
change('事件率使用事件数除以有效时长和候选频率，并将结果上限限制为3，用于表示单位候选周期内检测到的事件数量。','事件率使用事件数除以有效时长和候选频率，并将结果上限限制为3，用于表示单位候选周期内检测到的事件数量。频率微调、至少五个事件、偏移参照和事件率上限均为本文的经验规则，文献{[}22,25{]}不能证明这些具体参数已得到独立验证。')
change('除周期描述外，本文在每个频带补充五项包络形状统计。','除周期描述外，本文在每个频带补充五项包络形状统计。以峭度、峰值和均方根描述振动幅值与冲击程度具有既有研究基础{[}1{]}；这些指标在这里是对包络而非原始波形计算，具体变换由本文设置。')
change('实现分别对峭度和峰值因子使用log(1+x)变换，以缓和极端值的影响。','顶部1\\%能量比例是本文依据集中度思路采用的分位统计，文献{[}1{]}不直接给出该阈值。实现分别对峭度和峰值因子使用对数加一变换，以缓和极端值的影响；这两个变换是工程选择，尚未作独立消融。')
change('不同频带之间的能量分布通过对数相对均方根表示。设第b个频带的包络均方根为q\\_b，则该描述为log{[}q\\_b/(Σq\\_b+\\ensuremath{\\epsilon}){]}。谱平坦度则用非直流包络功率谱的几何均值除以算术均值，并限制在0至1之间。',r'''不同频带之间的能量分布通过对数相对均方根表示。设第$b$个频带的包络均方根为$q_b$，数值稳定常数为$\epsilon$，则该特征为：
\[
r_b=\log\!\left(\max\!\left[\frac{q_b}{\sum_{j=1}^{4}q_j+\epsilon},10^{-8}\right]\right)
\]
均方根是常见振动统计量{[}1{]}，但跨频带归一化及对数变换是本文的具体组合。谱平坦度用非直流包络功率谱的几何均值除以算术均值{[}23{]}，本文再将其限制在0至1之间；文献{[}23{]}来自音频信号处理，仅支持该统计量的定义，不证明其在本文轴承数据上独立有效。''')
change('不同物理描述的数值范围差异较大，直接输入多层感知机会增加优化难度。','不同物理描述的数值范围差异较大，直接输入多层感知机会增加优化难度。以中位数中心化、四分位距缩放属于常用稳健标准化方式{[}24{]}；本文在此基础上增加尺度下限和截断。')
change('中心和尺度均仅由当前配置的训练子集拟合。','式中的0.05尺度下限和正负8截断范围是本文实现选择，文献{[}24{]}不对这些数值的适用性作保证。中心和尺度均仅由当前配置的训练子集拟合。')
change('若窗口信号功率为$P_{s}$，设目标信噪比为S dB，则噪声功率按$P_{n}$=$P_{s}$/10\\^{}(S/10)确定。',r'''若窗口信号功率为$P_s$，目标信噪比为$S$ dB，则噪声功率为：
\[
P_n=P_s\,10^{-S/10}
\]''')
change('设测试集中正确分类窗口数为N\\_correct，总窗口数为N，准确率定义为ACC=N\\_correct/N。ACC反映总体窗口正确比例，',r'''设测试集中正确分类窗口数为$N_{\mathrm{correct}}$，总窗口数为$N$，准确率定义为：
\[
\mathrm{ACC}=\frac{N_{\mathrm{correct}}}{N}
\]
ACC反映总体窗口正确比例，''')
change('设第k类真正例和假负例分别为T$P_{k}$与F$N_{k}$，则该类召回率为Recal$l_{k}$=T$P_{k}$/(T$P_{k}$+F$N_{k}$)。四分类均衡准确率为四类召回率的算术平均：',r'''设第$k$类真正例和假负例分别为$TP_k$与$FN_k$，则该类召回率为：
\[
\mathrm{Recall}_k=\frac{TP_k}{TP_k+FN_k}
\]
四分类均衡准确率为四类召回率的算术平均：''')
change('第k类精确率为Precisio$n_{k}$=T$P_{k}$/(T$P_{k}$+F$P_{k}$)。F1为2×Precisio$n_{k}$×Recal$l_{k}$/(Precisio$n_{k}$+Recal$l_{k}$)。Macro-F1对四类F1做等权平均，',r'''设$FP_k$为第$k$类假正例，该类精确率和F1分别为：
\[
\mathrm{Precision}_k=\frac{TP_k}{TP_k+FP_k}
\]
\[
F1_k=\frac{2\,\mathrm{Precision}_k\,\mathrm{Recall}_k}{\mathrm{Precision}_k+\mathrm{Recall}_k}
\]
Macro-F1对四类F1做等权平均，''')

refs=r'''
{[}20{]} Duan J, Shi T, Zhou H, Xuan J, Zhang Y. Multiband envelope spectra extraction for fault diagnosis of rolling element bearings. Sensors, 2018, 18(5): 1466. DOI: 10.3390/s18051466.

{[}21{]} Moshrefzadeh A, Fasana A. The Autogram: An effective approach for selecting the optimal demodulation band in rolling element bearings diagnosis. Mechanical Systems and Signal Processing, 2018, 105: 294--318. DOI: 10.1016/j.ymssp.2017.12.009.

{[}22{]} Leahy D A, Elsner R F, Weisskopf M C. On searches for periodic pulsed emission: The Rayleigh test compared to epoch folding. The Astrophysical Journal, 1983, 272: 256--258. DOI: 10.1086/161288.

{[}23{]} Bardeli R. Source separation using the spectral flatness measure. CHiME Workshop on Machine Listening in Multisource Environments, 2011: 80--84. \url{https://www.isca-archive.org/chime_2011/bardeli11_chime.pdf}.

{[}24{]} scikit-learn developers. RobustScaler: Scale features using statistics that are robust to outliers. \url{https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.RobustScaler.html}. Accessed 2026-10-31.

{[}25{]} Rousseeuw P J, Croux C. Alternatives to the median absolute deviation. Journal of the American Statistical Association, 1993, 88(424): 1273--1283. DOI: 10.1080/01621459.1993.10476408.

{[}26{]} Antoni J. Cyclic spectral analysis of rolling-element bearing signals: Facts and fictions. Journal of Sound and Vibration, 2007, 304(3--5): 497--529. DOI: 10.1016/j.jsv.2007.02.029.
'''
change('\n\\end{document}', '\n'+refs+'\n\\end{document}')
counter=[0]
def make_equation(match):
    counter[0]+=1
    return '\\begin{equation}\n\\label{eq:auto%02d}\n%s\n\\end{equation}'%(counter[0],match.group(1).strip())
s=re.sub(r'\\\[(.*?)\\\]',make_equation,s,flags=re.S)
assert counter[0]==20,counter[0]
(out/'thesis.tex').write_text(s,encoding='utf-8')
print('displayed numbered equations:',counter[0], 'citations added: 20-26')
