from pathlib import Path
import re, json, shutil

root = Path(r'D:\Project\MY draft')
src = root/'thesis_document'/'成果'/'V3_论文_中英文文献补充_20261031'
dst = root/'thesis_document'/'成果'/'V3_论文_近期文献修订_20261031'
s = (src/'thesis.tex').read_text(encoding='utf-8')
marker = r'\chapter*{参考文献}'
body, bibliography = s.split(marker, 1)
entries = {int(n):t for n,t in re.findall(r'\\noindent\\begingroup\\samepage \{\[\}(\d+)\{\]\} (.*?)\\par\\endgroup', bibliography, re.S)}
assert len(entries) == 49
body = re.sub(r'\\textsuperscript\{\[([\d,]+)\]\}', lambda m: ''.join('<<O'+i+'>>' for i in m.group(1).split(',')), body)

new = {
'C1': ('胡济民, 罗洋, 于亦浩, 等. 多源内部故障激励下深沟球轴承三维动态解析建模及振动分析[J]. 机械工程学报, 2025, 61(23): 96--107. DOI: 10.3901/JME.2025.23.096.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.2025.23.096', '中文', 2025),
'C2': ('张春林, 吴允恒, 蔡克燊, 等. 基于改进连续小波变换增强非凸正则项稀疏分解的滚动轴承变转速故障冲击特征提取方法[J]. 机械工程学报, 2025, 61(1): 172--186. DOI: 10.3901/JME.2025.01.172.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.2025.01.172', '中文', 2025),
'C3': ('邵海东, 肖一鸣, 颜深, 等. 域泛化机械故障诊断研究进程与展望[J]. 中国科学: 技术科学, 2025, 55(1): 14--32. DOI: 10.1360/SST-2024-0149.', 'https://dds.sciengine.com/cfs/files/pdfs/1674-7259/E2EB8FAA7BE44851A0331B043AED032D.pdf', '中文', 2025),
'C4': ('董绍江, 夏宗佑, 邹松, 等. 数据缺失条件下基于对比嵌入特征生成的滚动轴承广义零样本故障诊断方法[J]. 机械工程学报, 2025, 61(23): 156--169. DOI: 10.3901/JME.2025.23.156.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.2025.23.156', '中文', 2025),
'C5': ('邵海东, 颜深, 刘政武, 等. 面向小样本场景的风机多增量故障诊断[J]. 机械工程学报, 2025, 61(10): 230--240. DOI: 10.3901/JME.2025.10.230.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.2025.10.230', '中文', 2025),
'C6': ('刘素艳, 董一林, 乔一鸣, 等. 基于改进蜣螂优化算法的轴承数字孪生模型的动态故障注入研究[J]. 机械工程学报, 2026, 62(3): 446--457. DOI: 10.3901/JME.260097.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.260097', '中文', 2026),
'C7': ('舒畅, 康建雄, 沈艳军, 等. 物理特征融合的半监督小样本滚动轴承故障诊断[J]. 工程科学学报, 2026, 48(7): 1605--1620. DOI: 10.13374/j.issn2095-9389.2025.11.17.001.', 'https://cje.ustb.edu.cn/article/doi/10.13374/j.issn2095-9389.2025.11.17.001?viewType=citedby-info', '中文', 2026),
'C8': ('胡文扬, 李奇, 林起鉴, 等. 面向旋转机械健康监测和早期故障检测的Diffusion-Ramanujan数字孪生架构[J]. 机械工程学报, 2026, 62(12): 77--86. DOI: 10.3901/JME.260488.', 'https://qikan.cmes.org/jxgcxb/CN/10.3901/JME.260488', '中文', 2026),
'E1': ('Qin Y, Zhao L, Chen Y, et al. Learnable wavelet-driven physically interpretable networks for bearing fault diagnosis under variable speed[J]. Mechanical Systems and Signal Processing, 2025, 237: 113121. DOI: 10.1016/j.ymssp.2025.113121.', 'https://www.sciencedirect.com/science/article/abs/pii/S0888327025008222', '英文', 2025),
'E2': ('Jia W, Dong Z, Shi H, et al. A zero-shot bearing fault diagnosis framework utilizing spatial relationships among primary features under variable working conditions[J]. Mechanical Systems and Signal Processing, 2025, 238: 113103. DOI: 10.1016/j.ymssp.2025.113103.', 'https://www.sciencedirect.com/science/article/pii/S0888327025008040', '英文', 2025),
'E3': ('Koutsoupakis J, Giagopoulos D. A simulation-enhanced deep learning framework for bearing fault diagnosis using an improved bearing force model[J]. Mechanical Systems and Signal Processing, 2025, 241: 113519. DOI: 10.1016/j.ymssp.2025.113519.', 'https://www.sciencedirect.com/science/article/pii/S0888327025012208', '英文', 2025),
'E4': ('Ma Z, Fu L, Tan D, et al. A mechanism-based sample generation method utilizing morphological analysis and dynamic modeling for online monitoring of rotor systems[J]. Mechanical Systems and Signal Processing, 2025, 228: 112438. DOI: 10.1016/j.ymssp.2025.112438.', 'https://www.sciencedirect.com/science/article/abs/pii/S0888327025001396', '英文', 2025),
'E5': ('Sun X, Xing J, Cong X, et al. MILT: A latent temporal framework for domain generalization in bearing fault diagnosis[J]. Mechanical Systems and Signal Processing, 2026, 260: 114919. DOI: 10.1016/j.ymssp.2026.114919.', 'https://www.sciencedirect.com/science/article/pii/S0888327026010769', '英文', 2026),
'E6': ('Wang T, Wang Y, Ji Y, et al. Progressive cross-model distillation with heterogeneous synergistic classifier for few-shot bearing fault diagnosis[J]. Mechanical Systems and Signal Processing, 2026, 260: 114991. DOI: 10.1016/j.ymssp.2026.114991.', 'https://www.sciencedirect.com/science/article/abs/pii/S0888327026011489', '英文', 2026),
'E7': ('Xv J, Zheng Y, Liao Z, et al. A fault features blind separation and extraction technique for weak multi-fault detection in rolling bearings[J]. Mechanical Systems and Signal Processing, 2025, 237: 113075. DOI: 10.1016/j.ymssp.2025.113075.', 'https://www.sciencedirect.com/science/article/pii/S0888327025007769', '英文', 2025),
}

def replace(a,b):
    global body
    if body.count(a)!=1: raise ValueError((body.count(a),a))
    body=body.replace(a,b,1)

replace(' 贾艳秋等从轴承故障类型、故障频率计算和包络分析出发讨论了诊断方法<<O2>>。', ' 胡济民等通过轴承动态建模分析内部故障激励下的振动响应<<C1>>，为理解故障激励与观测信号的关系提供了近期研究实例。')
replace('黄致远等将经验小波分解、独立分量分析与包络谱相结合，研究了故障特征的提取<<O10>>。', '张春林等结合小波变换和稀疏分解提取变转速下的故障冲击<<C2>>；Xv等利用循环频率估计与冲击分析研究弱复合故障的分离<<E7>>。')
replace('郭家昕等在滚动轴承诊断中引入核变换改进多线性主成分分析网络<<O12>>；郭俊锋等', '郭俊锋等')
replace('陈凯等结合深度特征提取与对抗域适应处理变工况数据<<O20>>；曹洁等', '曹洁等')
replace('徐倩楠和张永林使用动态分布自适应处理源域与目标域的分布差异<<O22>>，杨康等则研究了源域与目标域的子域对齐<<O23>>。', '杨康等研究了源域与目标域的子域对齐<<O23>>。邵海东等综述了域适应与域泛化的目标域信息条件及其不同任务设定<<C3>>。Sun等利用时域与频域表示及跨域对比学习研究未见工况诊断<<E5>>。')
replace('林伟等将变分模态分解与支持向量机结合用于小样本轴承识别<<O32>>；郭海宇等', '郭海宇等')
replace('这些方案的源数据和目标数据使用条件不同，不能仅按目标训练样本数比较标注成本。', '刘素艳等利用轴承动力学模型和故障注入生成样本<<C6>>，Ma等则借助形态分析和动力学建模生成转子状态样本<<E4>>。这些方案的源数据和目标数据使用条件不同，不能仅按目标训练样本数比较标注成本。')
replace('不同方法对预训练、样本构造和验证资源的要求各异，需要结合实验协议比较。', '邵海东等研究了小样本条件下风机故障的增量诊断<<C5>>；Wang等使用跨模态知识蒸馏和协同分类研究小样本轴承诊断<<E6>>。不同方法对预训练、样本构造和验证资源的要求各异，需要结合实验协议比较。')
replace('这些方法的目标域信息使用方式应与本文的单源直接测试区分。', '这些方法的目标域信息使用方式应与本文的单源直接测试区分。')
replace('生成质量和生成模型本身所需的训练数据，应与最终分类效果共同评估。', '生成质量和生成模型本身所需的训练数据，应与最终分类效果共同评估。针对训练中缺失故障类别的问题，董绍江等利用对比嵌入及生成模型研究广义零样本诊断<<C4>>；Jia等利用基本特征之间的关系研究变工况零样本识别<<E2>>。零样本任务与本文已知类别的小训练预算任务不同。')
replace('总体而言，物理引导和门控融合均已有较充分研究。', '近期研究进一步结合了物理知识与学习模型。Qin等构造可学习小波网络，将转速相关信息用于变转速轴承诊断<<E1>>；Koutsoupakis和Giagopoulos利用轴承受力模型生成数据，以补充深度学习训练<<E3>>。舒畅等从包络信号及包络谱中提取物理特征，并将其与半监督小样本诊断相结合<<C7>>。该研究使用标注与无标签数据，并引入分类权重的物理先验约束；本文使用训练子集中的标注窗口从头训练，将确定性物理描述通过门控残差形式引入分类模型。两者都涉及物理特征，不能将这一共同思路本身视为本文独创。胡文扬等将生成模型与周期变换结合，用于早期故障检测<<C8>>，其健康监测任务与本文闭集分类也有区别。\n\n总体而言，物理引导和门控融合均已有较充分研究。')

pat=re.compile(r'<<([OCE]\d+)>>')
order=[]
for k in pat.findall(body):
    if k not in order:order.append(k)
removed={2,10,12,20,22,32}
assert set(order)==({'O'+str(i) for i in entries if i not in removed}|set(new)), (set(new)-set(order))
number={k:i+1 for i,k in enumerate(order)}
body=re.sub(r'(?:<<[OCE]\d+>>)+',lambda m:r'\textsuperscript{['+','.join(str(number[k]) for k in pat.findall(m.group()))+']}',body)
out=[]
for k in order:
    entry=entries[int(k[1:])] if k.startswith('O') else new[k][0]
    if not k.startswith('O'):
        entry=re.sub(r'DOI: (10\.\d{4,9}/[^\s]+)',lambda m:r'DOI: \url{'+m.group(1).rstrip('.')+'}.',entry)
    out.append(r'\noindent\begingroup\samepage {[}'+str(number[k])+r'{]} '+entry+r'\par\endgroup')
final=body+marker+r'\markboth{参考文献}{参考文献}'+r'\addcontentsline{toc}{chapter}{参考文献}\label{ux53c2ux8003ux6587ux732e}'+'\n\n'+'\n\n'.join(out)+'\n\n'+r'\end{document}'+'\n'

def blocks(x):return re.findall(r'\\begin\{(?:equation|align|gather)\}.*?\\end\{(?:equation|align|gather)\}',x,re.S)
assert blocks(s)==blocks(final)
assert re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}',s)==re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}',final)
dst.mkdir(exist_ok=True,parents=True)
shutil.copytree(src/'media',dst/'media',dirs_exist_ok=True)
for p in src.glob('*.csv'):shutil.copy2(p,dst/p.name)
(dst/'thesis.tex').write_text(final,encoding='utf-8')
physical=(src/'物理特征文献依据对照.md').read_text(encoding='utf-8')
physical=re.sub(r'\[(\d+)\]',lambda m:'['+str(number['O'+m.group(1)])+']',physical)
physical+='\n近期补充研究表明，物理特征与小样本学习的结合已有先例，详见正文文献 ['+str(number['C7'])+']；该文不构成本文各特征具体参数的来源。\n'
(dst/'物理特征文献依据对照.md').write_text(physical,encoding='utf-8')
prior=(src/'文献检索与引用核验.md').read_text(encoding='utf-8')
priorrows={int(n):rest for n,rest in re.findall(r'^\| (\d+) \| (.*)$',prior,re.M)}
audit=['# 近期文献修订与引用核验','','截至 2026 年 10 月，以正式发表年份 2025、2026 统计近两年；期刊分区会随评价体系与年份变化，不将期刊名直接当作确定的中科院分区。新增条目优先核对期刊出版页面、DOI、题录及摘要；未能取得全文的条目，其说明以摘要及题录为限。','','| 编号 | 语言 | 核验来源 | 文献信息 |','|---|---|---|---|']
for k in order:
    if k.startswith('O'):
        audit.append('| '+str(number[k])+' | '+priorrows[int(k[1:])])
    else:
        e,u,lang,y=new[k]
        audit.append(f'| {number[k]} | {lang} | [期刊出版页面]({u}) | {e} |')
audit += ['', '本次移除原版中文文献 [2]、[10]、[12]、[20]、[22]、[32]，以更贴近主题的近期研究补足；保留较早的特征定义、统计与网络原始出处，避免用新近研究代替基础来源。', '', '部分中文期刊论文通过知网线索及期刊平台核验，不表述为全部逐篇获取知网全文。上述新研究并未在本文实验中复现，不作为相同实验条件下的性能比较。']
(dst/'文献检索与引用核验.md').write_text('\n'.join(audit)+'\n',encoding='utf-8')
(dst/'参考文献编号映射.json').write_text(json.dumps(number,ensure_ascii=False,indent=2),encoding='utf-8')
recent=[k for k in order if re.search(r', 202[56][,(：:]',entries[int(k[1:])] if k.startswith('O') else new[k][0])]
print('total',len(order),'new',len(new),'recent',len(recent),'recent keys',recent)
print('removed',removed)
