from pathlib import Path
from collections import Counter
import json
import re
import shutil

ROOT = Path(r'D:\Project\MY draft')
SOURCE = ROOT / 'thesis_document' / '成果' / 'V3_论文_语言修订_20261031'
TARGET = ROOT / 'thesis_document' / '成果' / 'V3_论文_中英文文献补充_20261031'
source = (SOURCE / 'thesis.tex').read_text(encoding='utf-8')
marker = '\\chapter*{参考文献}'
body, bibliography = source.split(marker, 1)
old = {int(n): v.strip() for n, v in re.findall(r'\{\[\}(\d+)\{\]\}\s*(.*?)(?=\n\s*\n|\\end\{document\})', bibliography, re.S)}
assert set(old) == set(range(1, 27)), sorted(old)

# Entries use the journal's own metadata where possible. Online-only entries
# preserve the CNKI record URL rather than inventing unavailable pagination.
refs = {
'C01': ('贾艳秋, 张兵, 陈雪梅. 滚动轴承的故障机理及诊断[J]. 化工装备技术, 2011, 32(4): 55--57.', 'https://cnki.istiz.org.cn/kcms/detail/detail.aspx?dbcode=CJFQ&dbname=CJFD2011&filename=HGZB201104018', '中文', '知网题录；卷页由可核验参考文献交叉核对'),
'C02': ('张良, 杨涛. 基于多重自相关与包络谱分析的滚动轴承故障诊断方法研究[J]. 组合机床与自动化加工技术, 2017(8): 93--96. DOI: 10.13462/j.cnki.mmtamt.2017.08.023.', 'https://qikan.cmes.org/zhjc/CN/10.13462/j.cnki.mmtamt.2017.08.023', '中文', '期刊出版页'),
'C03': ('张琛, 郭俊超, 甄冬, 等. 基于小波包时延相关解调的滚动轴承故障诊断方法[J]. 机械设计, 2020, 37(6): 24--28. DOI: 10.13841/j.cnki.jxsj.2020.06.004.', 'https://qikan.cmes.org/jxsj/CN/10.13841/j.cnki.jxsj.2020.06.004', '中文', '期刊原文和出版页'),
'C04': ('吴定会, 方钦, 吴楚宜. 基于数据生成与迁移学习的轴承小样本故障诊断[J]. 机械传动, 2020, 44(11): 139--144. DOI: 10.16578/j.issn.1004.2539.2020.11.023.', 'https://qikan.cmes.org/jxcd/CN/Y2020/V44/I11/139', '中文', '知网引文及期刊出版页'),
'C05': ('郭家昕, 程军圣, 杨宇. 改进多线性主成分分析网络及其在滚动轴承故障诊断中的应用[J]. 中国机械工程, 2022, 33(2): 187--193, 201. DOI: 10.3969/j.issn.1004-132X.2022.02.008.', 'https://qikan.cmes.org/zgjxgc/CN/10.3969/j.issn.1004-132X.2022.02.008', '中文', '期刊出版页'),
'C06': ('李可, 燕晗, 顾杰斐, 等. 基于shapelets时间序列的多源迁移学习滚动轴承故障诊断方法[J]. 中国机械工程, 2022, 33(24): 2990--2996, 3006. DOI: 10.3969/j.issn.1004-132X.2022.24.010.', 'https://qikan.cmes.org/zgjxgc/CN/10.3969/j.issn.1004-132X.2022.24.010', '中文', '期刊出版页'),
'C07': ('黄致远, 颜丙生, 刘兆亮. 基于FEWT-FastICA的滚动轴承故障特征识别方法[J/OL]. 机电工程, 2023(4)[2026-10-31]. \\url{https://cnki.istiz.org.cn/kcms/detail/detailall.aspx?dbcode=CJFD&dbname=CJFD2023&filename=jdgc202304005}.', 'https://cnki.istiz.org.cn/kcms/detail/detailall.aspx?dbcode=CJFD&dbname=CJFD2023&filename=jdgc202304005', '中文', '知网题录；页码未取得，按在线文献保留'),
'C08': ('林伟, 曹登学, 顾雨. 一种IMNGO-VMD小样本数据下的轴承故障识别方法[J]. 制造技术与机床, 2023(9): 17--24. DOI: 10.19287/j.mtmt.1005-2402.2023.09.002.', 'https://qikan.cmes.org/zzjsyjc/CN/10.19287/j.mtmt.1005-2402.2023.09.002', '中文', '期刊出版页'),
'C09': ('董绍江, 朱朋, 朱孙科, 等. 基于仿真数据驱动和领域自适应的滚动轴承故障诊断方法[J]. 中国机械工程, 2023, 34(6): 694--702. DOI: 10.3969/j.issn.1004-132X.2023.06.008.', 'https://qikan.cmes.org/zgjxgc/CN/10.3969/j.issn.1004-132X.2023.06.008', '中文', '期刊出版页'),
'C10': ('卞文彬, 邓艾东, 刘东川, 等. 基于改进深度残差收缩网络的风电机组滚动轴承故障诊断方法[J]. 机械工程学报, 2023, 59(12): 202--214. DOI: 10.3901/JME.2023.12.202.', 'https://qikan.cmes.org/jxgcxb/CN/Y2023/V59/I12', '中文', '期刊目录和摘要'),
'C11': ('陈凯, 张礼华, 赵恒, 等. 基于深度特征提取和对抗域适应网络的滚动轴承故障诊断[J]. 制造技术与机床, 2023(1): 9--15. DOI: 10.19287/j.mtmt.1005-2402.2023.01.001.', 'https://qikan.cmes.org/zzjsyjc/CN/10.19287/j.mtmt.1005-2402.2023.01.001', '中文', '期刊原文和出版页'),
'C12': ('曹洁, 尹浩楠, 雷晓刚, 等. 基于领域自适应的变工况轴承故障诊断[J]. 北京航空航天大学学报, 2024, 50(8): 2382--2390. DOI: 10.13700/j.bh.1001-5965.2022.0631.', 'https://www.spacejournal.cn/article/doi/10.13700/j.bh.1001-5965.2022.0631', '中文', '期刊出版页'),
'C13': ('杨岗, 徐五一, 邓琴, 等. 基于振动信号的滚动轴承复合故障诊断研究综述[J]. 西华大学学报（自然科学版）, 2024, 43(1): 48--69. DOI: 10.12198/j.issn.1673-159X.5096.', 'https://aj.xhu.edu.cn/xhdxxbzkb/cn/article/doi/10.12198/j.issn.1673-159X.5096', '中文', '期刊原文和出版页'),
'C14': ('郭海宇, 邹圣公, 张晓光, 等. 基于多源小波变换神经网络的旋转机械轴承故障诊断[J]. 中国机械工程, 2024, 35(11): 2026--2034. DOI: 10.3969/j.issn.1004-132X.2024.11.014.', 'https://qikan.cmes.org/zgjxgc/CN/10.3969/j.issn.1004-132X.2024.11.014', '中文', '期刊出版页'),
'C15': ('黄包裕, 张永祥. 一种基于三步法的轴承故障诊断方法[J]. 机械工程学报, 2024, 60(14): 51--68. DOI: 10.3901/JME.2024.14.051.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.2024.14.051', '中文', '期刊出版页'),
'C16': ('徐倩楠, 张永林. 不同工况下滚动轴承动态分布自适应故障诊断方法[J]. 制造业自动化, 2024, 46(11): 117--125. DOI: 10.3969/j.issn.1009-0134.2024.11.016.', 'https://qikan.cmes.org/zzyzdh/CN/Y2024/V46/I11', '中文', '期刊目录与摘要'),
'C17': ('秦毅, 杨瑞, 赵丽娟, 等. 面向轴承故障诊断的无监督噪声自适应匹配追踪算法展开去噪网络[J]. 机械工程学报, 2025, 61(12): 26--38. DOI: 10.3901/JME.2025.12.026.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.2025.12.026', '中文', '知网题录及期刊出版页'),
'C18': ('杨康, 陈学军, 张磊, 等. 基于联合子域对比对齐的轴承跨域故障诊断[J]. 中国机械工程, 2025, 36(5): 1065--1073. DOI: 10.3969/j.issn.1004-132X.2025.05.018.', 'https://cjournal.hep.com.cn/1004-132X/CN/1160171642459710292', '中文', '期刊原文和出版页'),
'C19': ('郭俊锋, 谭宝宏, 王智明. 基于MDAM-GhostCNN的滚动轴承故障诊断方法[J]. 北京航空航天大学学报, 2025, 51(4): 1172--1184. DOI: 10.13700/j.bh.1001-5965.2023.0224.', 'https://bhxb.buaa.edu.cn/bhzk/cn/article/doi/10.13700/j.bh.1001-5965.2023.0224', '中文', '期刊原文和出版页'),
'C20': ('王庆楠, 陈倩, 李涛, 等. 基于轻量化孪生网络的小样本轴承故障诊断[J]. 聊城大学学报（自然科学版）, 2026, 39(1): 11--23. DOI: 10.19728/j.issn1672-6634.2025010011.', 'https://cjournal.hep.com.cn/1672-6634/CN/10.19728/j.issn1672-6634.2025010011', '中文', '知网题录及期刊出版页'),
'E01': ('He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770--778. DOI: 10.1109/CVPR.2016.90.', 'https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html', '英文', '会议论文原文'),
'E02': ('Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2018: 7132--7141. DOI: 10.1109/CVPR.2018.00745.', 'https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html', '英文', '会议论文原文'),
'E03': ('Ismail Fawaz H, Forestier G, Weber J, Idoumghar L, Muller P A. Deep learning for time series classification: a review[J]. Data Mining and Knowledge Discovery, 2019, 33(4): 917--963. DOI: 10.1007/s10618-019-00619-1.', 'https://link.springer.com/article/10.1007/s10618-019-00619-1', '英文', '期刊与作者版本'),
}

def add_after(anchor, content):
    global body
    if body.count(anchor) != 1:
        raise ValueError(f'Insertion anchor has {body.count(anchor)} matches: {anchor}')
    body = body.replace(anchor, anchor + content, 1)

# Literature is placed with the claim it supports, rather than being added
# to experimental findings or presented as independent validation of V3.
add_after('轴承局部接触激发的冲击', '') if False else None
add_after('这些方法具有明确的工程含义，但在实际使用时仍需考虑共振带选择、参数误差和背景噪声等因素。', ' 贾艳秋等从轴承故障类型、故障频率计算和包络分析出发讨论了诊断方法<<C01>>。')
add_after('因而，传统信号分析在提供可解释描述的同时，也存在特征选取和阈值确定依赖经验的问题。', ' 张良和杨涛将多重自相关与包络谱结合以提取周期调制成分<<C02>>；张琛等使用小波包和时延相关解调提取强噪声背景下的冲击特征<<C03>>。黄致远等将经验小波分解、独立分量分析与包络谱相结合，研究了故障特征的提取<<C07>>。这些研究均体现了信号预处理和候选周期分析在轴承诊断中的作用，但具体分解方法与本文不同。')
add_after('参数量增加带来的性能变化，需要与输入信息和网络结构变化共同考虑。', ' 残差学习通过学习输入与输出之间的修正改善深层网络的优化<<E01>>。郭家昕等在滚动轴承诊断中引入核变换改进多线性主成分分析网络<<C05>>；郭俊锋等结合混合域注意力与轻量卷积进行故障识别<<C19>>。时序分类研究还系统比较了多种深度网络的建模方式<<E03>>。这些结构所需的输入表示和参数量并不相同。')
add_after('其抗噪能力需要在具体噪声类型和强度下验证。', ' 卞文彬等在残差收缩网络中加入稠密连接，研究强噪声及不同负载下的轴承诊断<<C10>>。黄包裕和张永祥提出三步诊断方法，以抑制传递路径效应和背景噪声<<C15>>；秦毅等将噪声估计与算法展开网络结合，用于去噪和故障诊断<<C17>>。这些方法分别从网络结构或信号处理环节应对干扰，不能直接证明本文固定物理特征对任意噪声均有效。')
add_after('比较评价结果时，不能忽略目标域访问权限的差别。', ' 国内研究也采用了不同的域适应策略。陈凯等结合深度特征提取与对抗域适应处理变工况数据<<C11>>；曹洁等采用局部分布对齐并考虑类别不平衡<<C12>>。徐倩楠和张永林使用动态分布自适应处理源域与目标域的分布差异<<C16>>，杨康等则研究了源域与目标域的子域对齐<<C18>>。这些方法的目标域信息使用方式应与本文的单源直接测试区分。')
add_after('生成质量和生成模型本身所需的训练数据，应与最终分类效果共同评估。', ' 吴定会等结合数据生成与迁移学习，利用源模型和少量目标数据开展轴承故障诊断<<C04>>；李可等使用多个来源域和时间序列形状信息进行迁移<<C06>>。董绍江等通过物理模型生成仿真数据，再结合领域自适应研究真实轴承诊断<<C09>>。这些方案的源数据和目标数据使用条件不同，不能仅按目标训练样本数比较标注成本。')
add_after('这些研究说明', '') if False else None
add_after('模型仍然需要独立划出的来源验证窗口进行逐轮选择；当训练预算很低时，验证窗口数量甚至可能超过训练窗口数量。', ' 林伟等将变分模态分解与支持向量机结合用于小样本轴承识别<<C08>>；郭海宇等研究了基于小波时频变换和卷积网络的小样本诊断<<C14>>。王庆楠等则采用轻量化孪生网络构造样本对<<C20>>。不同方法对预训练、样本构造和验证资源的要求各异，需要结合实验协议比较。')
add_after('故障特征不一定表现为与运动学频率完全重合的单一谱线，而可能以调制、谐波、边带和宽频瞬态等形式出现。', ' 杨岗等从振动信号处理与智能诊断两个方面综述了滚动轴承复合故障诊断<<C13>>，也反映出不同故障成分之间的分离与辨识具有难度。')
add_after('将权重与原通道逐元素相乘，使模型能够根据当前窗口调整不同通道的响应强度。', ' 这种通道重标定结构可参考Squeeze-and-Excitation网络的设计思路<<E02>>；本文采用的通道维数和数据编码结构由当前模型确定。')
add_after('实验使用凯斯西储大学轴承数据中心的振动数据，研究对象包括正常、内圈故障、外圈故障和滚动体故障四类状态。', ' 数据及台架说明见CWRU轴承数据中心<<O19>>。')
add_after('Moshrefzadeh和Fasana利用包络相关的自相关结构分析轴承故障的周期冲击{[}21{]}。', ' 张良和杨涛也利用自相关与包络分析识别周期调制成分<<C02>>。')

# Convert existing hand-written citation tags to stable identifiers.
def expand_old(match):
    ids = []
    for p in match.group(1).split(','):
        p = p.strip()
        if '-' in p:
            a, b = map(int, p.split('-'))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(p))
    return ''.join(f'<<O{i}>>' for i in ids)
body = re.sub(r'\{\[\}([0-9,\-]+)\{\]\}', expand_old, body)

# Automatically assign reference numbers by first appearance, including all
# entries in an existing range, then render each cluster as one superscript.
key_pat = re.compile(r'<<([OCE]\d+)>>')
order = []
for k in key_pat.findall(body):
    if k not in order:
        order.append(k)
expected = {f'O{i}' for i in range(1, 27)} | set(refs)
if set(order) != expected:
    raise ValueError(f'Citation mismatch: missing={expected-set(order)} extra={set(order)-expected}')
number = {k:i+1 for i,k in enumerate(order)}
def render_cluster(m):
    keys = key_pat.findall(m.group(0))
    return '\\textsuperscript{[' + ','.join(str(number[k]) for k in keys) + ']}'
body = re.sub(r'(?:<<[OCE]\d+>>)+', render_cluster, body)

def format_old(i):
    value = old[i].replace('\n', ' ')
    if i == 19:
        return value.replace('apparatus description.', 'apparatus description[EB/OL].', 1)
    if i == 23:
        return value.replace('spectral flatness measure.', 'spectral flatness measure[C].', 1)
    if i == 24:
        return value.replace('robust to outliers.', 'robust to outliers[EB/OL].', 1)
    parts = value.split('. ', 2)
    assert len(parts) == 3, (i, value)
    return parts[0] + '. ' + parts[1] + '[J]. ' + parts[2]

entries = []
for key in order:
    entry = format_old(int(key[1:])) if key.startswith('O') else refs[key][0]
    # DOI addresses can break across lines; keeping each entry together avoids
    # leaving a reference number on one page and its details on the next.
    entry = re.sub(r'DOI: (10\.\d{4,9}/[^\s]+)',
                   lambda m: 'DOI: \\url{' + m.group(1).rstrip('.') + '}.'
                   if m.group(1).endswith('.') else 'DOI: \\url{' + m.group(1) + '}', entry)
    entries.append(f'\\noindent\\begingroup\\samepage {{[}}{number[key]}{{]}} {entry}\\par\\endgroup')
final = body + marker + '\\addcontentsline{toc}{chapter}{参考文献}\\label{ux53c2ux8003ux6587ux732e}\n\n' + '\n\n'.join(entries) + '\n\n\\end{document}\n'

# Ensure the science and display structure are unchanged outside citations.
def formula_blocks(x):
    return re.findall(r'\\begin\{(?:equation|align|gather)\}.*?\\end\{(?:equation|align|gather)\}', x, re.S)
assert formula_blocks(source) == formula_blocks(final)
def commands(x):
    return re.findall(r'\\(?:begin|end|chapter\*?|section|subsection|label|ref|eqref|includegraphics)(?:\[[^\]]*\])?\{[^}]*\}', x)
assert commands(source) == commands(final)

TARGET.mkdir(parents=True, exist_ok=True)
shutil.copytree(SOURCE/'media', TARGET/'media', dirs_exist_ok=True)
for filename in ['boxplot_seed_values.csv', 'class_recall_sd_values.csv', 'cost_sd_values.csv']:
    shutil.copy2(SOURCE/filename, TARGET/filename)
(TARGET/'thesis.tex').write_text(final, encoding='utf-8')

map_text = (SOURCE/'物理特征文献依据对照.md').read_text(encoding='utf-8')
map_text = re.sub(r'\[(\d+)\]', lambda m:'['+str(number['O'+m.group(1)])+']', map_text)
map_text += '\n\n本版按正文首次出现顺序重新编号，并补充中文研究；本表仍只说明各文献的相关依据。\n'
(TARGET/'物理特征文献依据对照.md').write_text(map_text, encoding='utf-8')

audit = ['# 文献检索与引用核验', '', '文献编号按本版正文首次出现顺序排列。知网可访问页面提供题录、摘要；其余使用期刊或会议原始出版页面核对。文献摘要可以支持研究内容的概括，但不能代替逐页全文核验。', '', '| 新编号 | 语言 | 来源或核验方式 | 题录 |', '|---|---|---|---|']
for key in order:
    if key in refs:
        entry, url, lang, how = refs[key]
        plain = re.sub(r'\\url\{[^}]*\}', '', entry).replace('\\', '')
        audit.append(f'| {number[key]} | {lang} | [{how}]({url}) | {plain} |')
    else:
        i=int(key[1:]);lang='中文' if i==11 else '英文'
        audit.append(f'| {number[key]} | {lang} | 原论文保留条目 O{i} | {old[i].replace(chr(10), " ")} |')
audit += ['', '原论文英文期刊及会议 DOI 经 Crossref 检索 22 项；中文原文献由期刊页面核验。CWRU 数据源、谱平坦度会议论文和软件文档按其各自来源保留。新增中文 C07 在知网题录中未取得页码，暂按在线期刊条目列示，不臆填。', '', '部分中文文献通过知网节点直接检索，另一些通过知网线索和期刊出版页面核验；不能将所有条目表述为已经逐篇下载知网全文。新增文献用于综述与方法依据，尚未在本文实验中复现为对比模型。']
(TARGET/'文献检索与引用核验.md').write_text('\n'.join(audit)+'\n', encoding='utf-8')
(TARGET/'参考文献编号映射.json').write_text(json.dumps(number, ensure_ascii=False, indent=2), encoding='utf-8')
print('total', len(order), 'Chinese', 1+sum(k.startswith('C') for k in order), 'English', sum(k.startswith('E') or (k.startswith('O') and k!='O11') for k in order))
print('English academic papers', sum(k.startswith('E') or (k.startswith('O') and k not in ['O11','O19','O24']) for k in order))
