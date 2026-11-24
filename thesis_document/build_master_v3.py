from pathlib import Path
import ast, re, sys
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

R=Path(r'D:\Project\MY draft')
SRC=R/'thesis_document'
OUT=R/'thesis_document'/'成果'/'V3滚动轴承物理引导故障诊断_硕士论文正文稿.docx'
RESULT=R/'experiments/CWRU/实验结果'/'CWRU_V3'

def font(run,sz=12,bold=False,east='宋体'):
    run.font.name='Times New Roman'; run.font.size=Pt(sz); run.bold=bold
    run.font.color.rgb=RGBColor(0,0,0); run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),east)

def paragraph(doc,s,indent=True):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent=Pt(24) if indent else Pt(0)
    p.paragraph_format.line_spacing=1.30; p.paragraph_format.space_after=Pt(4)
    font(p.add_run(s),12); return p

def heading(doc,s,level):
    p=doc.add_heading(s,level=level); p.paragraph_format.keep_with_next=True
    if level==1: p.paragraph_format.page_break_before=True; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs: font(r,16 if level==1 else 14 if level==2 else 12,True,'黑体')
    return p

def field(p,instruction):
    r=p.add_run(); f=OxmlElement('w:fldChar'); f.set(qn('w:fldCharType'),'begin'); r._r.append(f)
    t=OxmlElement('w:instrText'); t.set(qn('xml:space'),'preserve'); t.text=instruction; r._r.append(t)
    e=OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'),'end'); r._r.append(e)

def fig(doc,path,caption,width=5.8):
    path=R/path
    if not path.exists(): raise FileNotFoundError(path)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
    p.paragraph_format.first_line_indent=Pt(0); p.add_run().add_picture(str(path),width=Inches(width))
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Pt(0)
    p.paragraph_format.space_after=Pt(6); font(p.add_run(caption),10.5)

def tbl(doc,caption,headers,rows):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
    font(p.add_run(caption),10.5)
    table=doc.add_table(rows=1,cols=len(headers)); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=True
    for j,x in enumerate(headers): table.cell(0,j).text=str(x)
    trpr=table.rows[0]._tr.get_or_add_trPr(); repeat=OxmlElement('w:tblHeader'); repeat.set(qn('w:val'),'true'); trpr.append(repeat)
    for row in rows:
        cells=table.add_row().cells
        for j,x in enumerate(row): cells[j].text=str(x)
    for i,row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
            for edge in ['top','bottom','left','right']:
                el=OxmlElement('w:'+edge)
                if (edge=='top' and i==0) or (edge=='bottom' and i in [0,len(table.rows)-1]):
                    el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'8' if edge=='top' or i else '5'); el.set(qn('w:color'),'000000')
                else: el.set(qn('w:val'),'nil')
                borders.append(el)
            pr.append(borders)
            for p in cell.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Pt(0); p.paragraph_format.space_after=Pt(0)
                for r in p.runs: font(r,9.5,i==0)

def equation(doc,s):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Pt(0)
    om=OxmlElement('m:oMathPara'); math=OxmlElement('m:oMath'); mr=OxmlElement('m:r'); mt=OxmlElement('m:t'); mt.text=s
    mr.append(mt); math.append(mr); om.append(math); p._p.append(om)

def reference_list():
    tree=ast.parse((SRC/'build_thesis_document.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='refs' for t in node.targets): return ast.literal_eval(node.value)
    raise RuntimeError('references missing')

def result_table(doc,kind):
    if kind=='selection':
        import json
        p=RESULT/'run_20261022_2105'/'selection_evaluation'/'selection.json'
        rows=[[f"{r['aux_weight']:.2f}",f"{r['mean_best_validation_ce']:.4g}",f"{r['sample_sd']:.4g}"] for r in json.loads(p.read_text(encoding='utf-8'))['candidates']]
        tbl(doc,'表5-1 辅助系数候选值的五种子验证交叉熵',['系数λ','均值','样本标准差'],rows); return
    if kind=='cost':
        frame=pd.read_csv(RESULT/'load_small_20261025_2017'/'computational_cost.csv').groupby('variant').agg(params=('parameters','mean'),latency=('latency_ms','mean'),mem=('peak_gpu_mib','mean'))
        rows=[[name,f"{frame.loc[v,'params']:.0f}",f"{frame.loc[v,'latency']:.2f}",f"{frame.loc[v,'mem']:.2f}"] for v,name in [('baseline_data','数据单分支'),('data_aux_ce','融合方案')]]
        tbl(doc,'表5-5 小样本跨负载模型的计算成本',['方法','参数量','推理延迟/ms','峰值显存/MiB'],rows); return
    path={'window':'window_random_20261023_0853/evaluation/between_subset_summary.csv','full':'load_full_20261024_0950/evaluation/transfer_equal_direction_mean_std.csv','small':'load_small_20261025_2017/evaluation/transfer_between_subset_summary.csv'}[kind]
    df=pd.read_csv(RESULT/path); ratios=[.1,.2,.5] if kind!='full' else [1.0]
    conditions=[('clean_only','clean','纯干净/干净'),('awgn_mixed','clean','AWGN/干净'),('awgn_mixed','white_-4','AWGN/−4 dB')]
    rows=[]
    for ratio in ratios:
        for protocol,condition,label in conditions:
            sub=df[(df.ratio==ratio)&(df.protocol==protocol)&(df.condition==condition)]
            b=sub[sub.variant=='baseline_data'].iloc[0]; f=sub[sub.variant=='data_aux_ce'].iloc[0]
            rows.append([f'{ratio:.0%}',label,f'{b.balanced_acc_mean*100:.2f}±{b.balanced_acc_std*100:.2f}',f'{f.balanced_acc_mean*100:.2f}±{f.balanced_acc_std*100:.2f}',f'{(f.balanced_acc_mean-b.balanced_acc_mean)*100:+.2f}'])
    titles={'window':'表5-2 随机窗口小样本代表性条件BA及子集间标准差','full':'表5-3 全样本跨负载12方向等权BA及五种子标准差','small':'表5-4 小样本跨负载12方向等权BA及子集间标准差'}
    tbl(doc,titles[kind],['训练比例','训练/测试','单分支BA/%','融合方案BA/%','差值/百分点'],rows)

def main():
    doc=Document(); sec=doc.sections[0]; sec.page_width=Cm(21); sec.page_height=Cm(29.7)
    sec.top_margin=Cm(2.5); sec.bottom_margin=Cm(2.4); sec.left_margin=Cm(2.8); sec.right_margin=Cm(2.5)
    st=doc.styles['Normal']; st.font.name='Times New Roman'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); st.font.size=Pt(12)
    for n in ['Title','Heading 1','Heading 2','Heading 3']:
        st=doc.styles[n]; st.font.color.rgb=RGBColor(0,0,0); st.font.name='Times New Roman'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体')
        ppr=st._element.get_or_add_pPr(); b=ppr.find(qn('w:pBdr'))
        if b is not None: ppr.remove(b)
    footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER; field(footer,' PAGE ')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(64)
    font(p.add_run('硕士学位论文正文'),22,True,'黑体')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(38)
    font(p.add_run('面向小样本与跨负载工况的\n物理引导双流融合轴承故障诊断研究'),19,True,'黑体')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(28)
    font(p.add_run('基于 CWRU V3 固定实验的研究'),12)
    doc.add_page_break()
    heading(doc,'摘  要',2)
    paragraph(doc,'滚动轴承故障诊断在实际应用中同时面临训练标签有限、振动信号受噪声干扰和运行负载变化等问题。仅依靠原始波形训练的深度网络在样本充足时能够学习判别特征，但在来源数据较少且目标工况变化时，其性能和训练稳定性可能下降。针对这一问题，本文研究一种融合原始振动数据表示和确定性物理描述的双流门控诊断方法，并在记录内随机窗口识别及单源跨负载条件下开展系统评价。')
    paragraph(doc,'首先，根据轴承局部损伤激发周期冲击和结构共振的机理，在四个频带内围绕转频及轴承候选故障阶次构造谐波、重复性、相位和形状描述，形成212维物理特征。物理特征的稳健中心与尺度仅使用相应训练子集拟合。其次，采用扩张残差卷积提取原始振动数据表示，将物理描述映射至相同维数，并通过逐通道门控与残差加和形成融合表示。训练阶段引入数据分支辅助分类，共享分类头，辅助损失系数按照预设验证规则固定。')
    paragraph(doc,'实验使用CWRU数据集40条记录形成的687个不重叠窗口，设置纯干净和加性高斯白噪声混合训练，比较10%、20%、50%和100%训练预算。随机窗口与单源跨负载实验分别评价记录内窗口识别和负载变化下的直接迁移。小样本预算采用3个固定子集，每个配置使用5个训练种子。单源跨负载结果对12个迁移方向等权汇总。')
    paragraph(doc,'在AWGN训练和−4 dB测试条件下，记录内随机窗口10%预算的数据单分支与融合方案均衡准确率分别为82.66%和94.72%。全样本跨负载12方向等权结果分别为84.31%和89.51%。10%小样本跨负载结果分别为36.17%和85.96%。随着训练预算增加，平均差距缩小。方向和类别分析显示，融合方案并非在所有情况下占优，2 HP到1 HP方向及外圈召回存在局部退化。')
    paragraph(doc,'研究结果表明，在本文固定数据划分和实验条件下，物理引导融合与辅助分类构成的完整方案在低预算跨负载任务中具有较高的平均诊断性能。由于目前对照同时改变输入信息、网络容量、融合结构与训练目标，不能将结果单独归因于某一组成部分。记录依赖、目标负载开发曝光和人工白噪声设置也限制了结论的外推范围。')
    paragraph(doc,'关键词：滚动轴承；故障诊断；物理引导；小样本；跨负载；门控融合',False)
    doc.add_page_break(); heading(doc,'Abstract',2)
    paragraph(doc,'Rolling bearing diagnosis under limited labeled data is challenged by measurement noise and changes in operating load. This thesis investigates a physics-guided dual-stream fusion model and evaluates it under record-internal window recognition and single-source cross-load transfer. A deterministic 212-dimensional description is constructed from multi-band envelope responses, candidate fault orders, periodicity, phase concentration and distributional statistics. Its robust scaling parameters are fitted only on the corresponding training subset. A dilated residual encoder extracts a data representation from raw vibration, while a physics branch projects the descriptor into a second representation. Channel-wise gated residual fusion combines the two representations, and a shared classification head provides auxiliary supervision during training.')
    paragraph(doc,'Experiments use 687 non-overlapping windows from 40 CWRU records, with clean and additive white Gaussian noise training protocols. Training budgets of 10%, 20%, 50% and 100% are investigated. The small-sample conditions use three fixed subsets and five training seeds. Under AWGN training and −4 dB testing, the balanced accuracy of the data-only and fusion models is 82.66% and 94.72%, respectively, at the 10% record-internal window budget. In cross-load evaluation averaged equally across 12 directions, the corresponding values are 84.31% and 89.51% with the full source training set, and 36.17% and 85.96% with the 10% source budget. The average advantage decreases as the training budget increases, and adverse changes remain in certain transfer directions and outer-race recall.')
    paragraph(doc,'The findings support an empirical advantage of the complete fusion and auxiliary-supervision scheme under the fixed experimental conditions. They do not separately identify the effects of physics features, rotational-speed metadata, capacity, the gate or auxiliary loss. Shared records, prior development exposure to target loads and artificial white noise limit external generalization.',True)
    paragraph(doc,'Key words: rolling bearings; fault diagnosis; physics-guided learning; limited samples; cross-load transfer; gated fusion',False)
    doc.add_page_break(); heading(doc,'目  录',2)
    paragraph(doc,'在 Word 中可更新目录及页码。',False)
    p=doc.add_paragraph(); field(p,' TOC \\o "1-1" \\h \\z \\u ')
    # Chapter text is parsed from independently verified writing drafts.
    sources=['masters_intro_lit.md','masters_methods.md','masters_results.md']
    alltext='\n\n'.join((SRC/x).read_text(encoding='utf-8') for x in sources)
    # Insert selected figures once, adjacent to relevant sections, without modifying frozen outputs.
    inserts=[
      (['损伤机理','故障机理','特征频率','振动基础'], 'thesis_document/figures/fig01_mechanism.png','图3-2 轴承局部损伤、振动响应与物理描述的关系'),
      (['物理描述','物理特征构造','物理特征提取'], 'thesis_document/figures/fig02_descriptor.png','图3-3 多频带物理描述构造与训练内标准化'),
      (['门控融合','门控残差','整体架构','融合模型'], 'thesis_document/figures/masters_architecture_correct.png','图3-1 双流门控残差融合模型结构'),
      (['实验框架','总体实验','实验流程','实验总体','阶段安排'], 'thesis_document/figures/fig04_experiment_flow.png','图4-1 实验划分和评价流程'),
      (['系数选择','系数选取','辅助损失系数'], 'thesis_document/v3_complete_figures/selection.png','图5-1 辅助损失系数的验证结果'),
      (['随机窗口小样本','随机窗口实验结果','随机窗口结果'], 'thesis_document/成果/V3训练比例对比图/Random_window_balanced_acc.png','图5-2 随机窗口不同训练比例下的均衡准确率'),
      (['跨负载小样本结果','小样本跨负载结果','小样本跨负载实验'], 'thesis_document/成果/V3训练比例对比图/Cross_load_balanced_acc.png','图5-5 不同来源训练比例下12方向等权均衡准确率'),
      (['方向差异','方向级','迁移方向'], 'thesis_document/figures/fig09_direction_heatmap.png','图5-3 全样本跨负载−4 dB方向级BA差值'),
      (['类别差异','类别召回','类别层面'], 'thesis_document/figures/fig11_class_recall.png','图5-4 小样本跨负载−4 dB类别召回率'),
      (['计算成本','计算复杂度','计算开销'], 'thesis_document/figures/fig13_cost.png','图5-6 参数量与推理延迟比较'),
    ]
    inserted=set(); pending=[]; current_chapter=0
    blocks=re.split(r'\n\s*\n',alltext)
    for block in blocks:
        block=block.strip()
        if not block: continue
        lines=block.splitlines()
        if len(lines)>1 and all(x.startswith('#') for x in lines):
            pieces=lines
        else: pieces=[block]
        for piece in pieces:
            piece=piece.strip()
            if not piece: continue
            if piece.startswith('#'):
                for i in pending:
                    _,path,cap=inserts[i]; fig(doc,path,cap); inserted.add(i)
                pending=[]
                first=piece.splitlines()[0]; n=len(first)-len(first.lstrip('#')); txt=first.lstrip('#').strip()
                heading(doc,txt,min(n,3))
                if n==2 and txt.startswith(('5.2 ','5.3 ','5.4 ','5.5 ','5.6 ')):
                    result_table(doc,{'5.2':'selection','5.3':'window','5.4':'full','5.5':'small','5.6':'cost'}[txt[:3]])
                if n==1:
                    m=re.search(r'第\s*([一二三四五六七0-9]+)\s*章',txt)
                    if m: current_chapter={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7}.get(m.group(1),int(m.group(1)) if m.group(1).isdigit() else current_chapter)
                for i,(keys,path,cap) in enumerate(inserts):
                    target=int(re.search(r'图(\d)',cap).group(1))
                    if i not in inserted and i not in pending and current_chapter==target and any(key in txt for key in keys): pending.append(i)
                rest='\n'.join(piece.splitlines()[1:]).strip()
                if rest: paragraph(doc,rest)
            elif piece.startswith('|'):
                rows=[[c.strip() for c in x.strip().strip('|').split('|')] for x in piece.splitlines() if x.strip().startswith('|')]
                rows=[x for x in rows if not all(re.fullmatch(r'[:\- ]+',z or '-') for z in x)]
                if len(rows)>1 and all(len(x)==len(rows[0]) for x in rows): tbl(doc,'实验结果比较',rows[0],rows[1:])
            elif piece.startswith('$$') and piece.endswith('$$'): equation(doc,piece.strip('$').strip())
            elif len(piece)<180 and '\n' not in piece and re.match(r'^(f_FTF|f_BPFO|f_BPFI|f_BSF|Rₕ|C\(f\)|ũⱼ|hᵢ₊₁|d|g|z|L|J_val|BA)\s*=',piece):
                equation(doc,piece.rstrip('。；'))
            else:
                piece=piece.replace('图3-1可用于展示','图3-1展示了')
                for line in piece.splitlines():
                    line=line.strip()
                    if line: paragraph(doc,line,not line.startswith(('- ','1. ','2. ','3. ')))
    for i in pending:
        _,path,cap=inserts[i]; fig(doc,path,cap); inserted.add(i)
    # If a section title differed, place missing visual evidence together rather than omit it.
    if len(inserted)<len(inserts):
        heading(doc,'附录 主要实验图表',1)
        for i,(_,path,cap) in enumerate(inserts):
            if i not in inserted: fig(doc,path,cap)
    heading(doc,'参考文献',1)
    for item in reference_list():
        item=item.replace('Wu C, Jiao G, Jin K, Zhang Y, Li Z, Zhang Z.', 'Wu C, Jiao G, Zhang Y, Jin K, Zhang Z, Li Z.')
        if item.startswith('[19]'): item+=' https://engineering.case.edu/bearingdatacenter/download-data-file.'
        paragraph(doc,item,False)
    doc.core_properties.author=''; doc.core_properties.last_modified_by=''; doc.core_properties.title='物理引导双流融合轴承故障诊断研究'
    doc.settings._element.append(OxmlElement('w:updateFields'))
    doc.save(OUT)
    print('output',OUT,'text_chars',len(re.sub(r'\s+','',alltext)),'figures',len(doc.inline_shapes),'paragraphs',len(doc.paragraphs),'tables',len(doc.tables))

if __name__=='__main__': main()
