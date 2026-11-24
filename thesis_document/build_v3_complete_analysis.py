from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(r'D:\Project\MY draft')
sys.path.insert(0, str(ROOT / 'thesis_document'))
from build_thesis_document import font, BLUE, RED, ORANGE, GREEN, DARK, LIGHT_GRID, blues, create_selection_figure, create_experiment_flow_figure

BASE = ROOT / 'experiments/CWRU/实验结果' / 'CWRU_V3'
WIN = BASE / 'window_random_20261023_0853' / 'evaluation'
FULL = BASE / 'load_full_20261024_0950' / 'evaluation'
SMALL = BASE / 'load_small_20261025_2017' / 'evaluation'
SEL = BASE / 'run_20261022_2105' / 'selection_evaluation'
FIG = ROOT / 'thesis_document' / 'v3_complete_figures'
OUT = ROOT / 'thesis_document' / '成果' / 'V3全部实验图表与逻辑分析.docx'
CONDITIONS = [('clean_only','clean','纯干净训练 / 干净'),('awgn_mixed','clean','AWGN训练 / 干净'),('awgn_mixed','white_-4','AWGN / -4 dB'),('awgn_mixed','white_-2','AWGN / -2 dB'),('awgn_mixed','white_+0','AWGN / 0 dB'),('awgn_mixed','white_+2','AWGN / +2 dB'),('awgn_mixed','white_+4','AWGN / +4 dB')]
RATIOS = [0.1,0.2,0.5,1.0]
VARIANTS = ['baseline_data','data_aux_ce']
NAMES = ['数据单分支','融合加辅助分类']

def save(im,name):
    p=FIG/name; im.save(p,dpi=(240,240)); return p

def text_font(r,size=9.5,bold=False):
    r.font.name='Times New Roman'; r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体'); r.font.size=Pt(size); r.bold=bold

def title(doc,s):
    p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    text_font(p.add_run(s),19,True)

def para(doc,s):
    p=doc.add_paragraph(); p.paragraph_format.first_line_indent=Pt(20); p.paragraph_format.space_after=Pt(4)
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; text_font(p.add_run(s),10)

def heading(doc,s,level=1):
    p=doc.add_heading(s,level=level); p.paragraph_format.keep_with_next=True
    for r in p.runs: text_font(r,14 if level==1 else 11,True); r.font.color.rgb=RGBColor(0,0,0)

def caption(doc,s):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(4); p.paragraph_format.keep_with_next=True
    text_font(p.add_run(s),9,True)

def table(doc,headers,rows,widths=None,fontsize=8.3):
    tab=doc.add_table(rows=1,cols=len(headers)); tab.alignment=WD_TABLE_ALIGNMENT.CENTER; tab.autofit=True
    for j,v in enumerate(headers): tab.rows[0].cells[j].text=str(v)
    trpr=tab.rows[0]._tr.get_or_add_trPr(); e=OxmlElement('w:tblHeader'); e.set(qn('w:val'),'true'); trpr.append(e)
    for row in rows:
        cells=tab.add_row().cells
        for j,v in enumerate(row): cells[j].text=str(v)
    for i,row in enumerate(tab.rows):
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for j,c in enumerate(row.cells):
            tcpr=c._tc.get_or_add_tcPr(); b=tcpr.first_child_found_in('w:tcBorders')
            if b is None: b=OxmlElement('w:tcBorders'); tcpr.append(b)
            for edge in ['top','bottom','left','right']:
                el=OxmlElement('w:'+edge)
                if edge=='top' and i==0 or edge=='bottom' and (i==0 or i==len(tab.rows)-1):
                    el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'8' if i!=0 or edge=='top' else '5'); el.set(qn('w:color'),'000000')
                else: el.set(qn('w:val'),'nil')
                b.append(el)
            for p in c.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.first_line_indent=Pt(0); p.paragraph_format.space_after=Pt(0)
                for r in p.runs: text_font(r,fontsize,i==0)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)

def fig(doc,path,cap,width=6.3):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
    p.paragraph_format.space_after=Pt(0); p.add_run().add_picture(str(path),width=Inches(width))
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(6); p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    text_font(p.add_run(cap),9)

def canvas_chart(data,ratio,name):
    img=Image.new('RGB',(1500,800),'white'); d=ImageDraw.Draw(img); left,right,top,bottom=160,1375,80,610
    for v in range(0,101,20):
        y=bottom-v/100*(bottom-top); d.line((left,y,right,y),fill=LIGHT_GRID,width=2); d.text((left-15,y),str(v),font=font(26,en=True),fill=DARK,anchor='rm')
    d.rectangle((left,top,right,bottom),outline=DARK,width=3)
    xs=np.linspace(left+60,right-60,5)
    for i,v in enumerate([-4,-2,0,2,4]): d.text((xs[i],bottom+20),f'{v:+d}',font=font(26,en=True),fill=DARK,anchor='mt')
    for variant,col in zip(VARIANTS,[BLUE,RED]):
        points=[]
        for idx,condition in enumerate(['white_-4','white_-2','white_+0','white_+2','white_+4']):
            r=data[(data.ratio==ratio)&(data.protocol=='awgn_mixed')&(data.variant==variant)&(data.condition==condition)]
            assert len(r)==1,(name,ratio,variant,condition,len(r))
            y=float(r.iloc[0]['balanced_acc_mean'])*100
            point=(xs[idx],bottom-y/100*(bottom-top)); points.append(point)
            d.ellipse((point[0]-8,point[1]-8,point[0]+8,point[1]+8),fill=col)
            if variant=='baseline_data':
                d.text((point[0],point[1]+15),f'{y:.1f}',font=font(23,en=True),fill=col,anchor='mt')
            else:
                d.text((point[0],point[1]-16),f'{y:.1f}',font=font(23,en=True),fill=col,anchor='mb')
        d.line(points,fill=col,width=5)
    d.text((770,675),'测试噪声信噪比 SNR（dB）',font=font(30,bold=True),fill=DARK,anchor='mm')
    d.text((50,48),'BA（%）',font=font(29,bold=True),fill=DARK,anchor='lt')
    for idx,(lab,col) in enumerate(zip(NAMES,[BLUE,RED])):
        x=195+idx*420; d.line((x,745,x+65,745),fill=col,width=6); d.text((x+80,745),lab,font=font(27),fill=DARK,anchor='lm')
    return save(img,name)

def selection_chart():
    candidates=json.loads((SEL/'selection.json').read_text(encoding='utf-8'))['candidates']
    im=Image.new('RGB',(1500,800),'white'); d=ImageDraw.Draw(im)
    left,right,top,bottom=170,1370,95,625
    def yy(v): return bottom-(v-4)/(10.2-4)*(bottom-top)
    for value in [4,5,6,7,8,9,10]:
        y=yy(value); d.line((left,y,right,y),fill=LIGHT_GRID,width=2)
        d.text((left-15,y),str(value),font=font(27,en=True),fill=DARK,anchor='rm')
    d.rectangle((left,top,right,bottom),outline=DARK,width=3)
    d.text((165,45),'最佳验证 CE（单位 10^-6）',font=font(30,bold=True),fill=DARK,anchor='lm')
    xs=np.linspace(left+80,right-80,5); pts=[]
    for i,c in enumerate(candidates):
        v=c['mean_best_validation_ce']*1e6; e=c['sample_sd']*1e6; x=xs[i]; y=yy(v); pts.append((x,y))
        d.line((x,yy(v-e),x,yy(v+e)),fill=DARK,width=3)
        d.line((x-10,yy(v-e),x+10,yy(v-e)),fill=DARK,width=3)
        d.line((x-10,yy(v+e),x+10,yy(v+e)),fill=DARK,width=3)
        d.text((x,bottom+22),f"{c['aux_weight']:.2f}",font=font(28,en=True),fill=DARK,anchor='mt')
        d.text((x,y-18),f'{v:.2f}',font=font(24,en=True),fill=RED if i==3 else BLUE,anchor='mb')
    d.line(pts,fill=BLUE,width=5)
    for i,(x,y) in enumerate(pts): d.ellipse((x-9,y-9,x+9,y+9),fill=RED if i==3 else BLUE)
    d.text((770,715),'辅助损失系数 λ',font=font(29,bold=True),fill=DARK,anchor='mm')
    return save(im,'selection.png')

def heat(data,ratio,protocol,condition,metric,name,by_subset=False):
    sub=data[(data.ratio==ratio)&(data.protocol==protocol)&(data.condition==condition)].copy()
    sub['direction']=sub['source_load_hp'].astype(str)+'→'+sub['target_load_hp'].astype(str)
    dirs=[f'{s}→{t}' for s in range(4) for t in range(4) if s!=t]
    groups=[4101,4102,4103] if by_subset else [0]
    arr=[]
    for direction in dirs:
        arr.append([float(sub[(sub.direction==direction)&(sub.subset==g)].iloc[0][metric])*100 for g in groups])
    arr=np.array(arr); im=Image.new('RGB',(1300,1100),'white'); d=ImageDraw.Draw(im)
    x0,y0=240,110; cw=900//len(groups); ch=70
    for j,g in enumerate(groups):
        d.text((x0+(j+.5)*cw,y0-30),f'子集 {g}' if by_subset else '五种子平均',font=font(28,bold=True),fill=DARK,anchor='mm')
    limit=max(6,float(np.quantile(np.abs(arr),.85)))
    for i,direction in enumerate(dirs):
        d.text((x0-20,y0+i*ch+ch/2),direction+' HP',font=font(28),fill=DARK,anchor='rm')
        for j,v in enumerate(arr[i]):
            strength=min(abs(v)/limit,1)
            color=tuple(int(255-(255-c)*strength) for c in (BLUE if v>=0 else RED))
            box=(x0+j*cw,y0+i*ch,x0+(j+1)*cw,y0+(i+1)*ch)
            d.rectangle(box,fill=color,outline='white',width=3)
            d.text(((box[0]+box[2])/2,(box[1]+box[3])/2),f'{v:+.2f}',font=font(26,bold=True,en=True),fill='white' if strength>.62 else DARK,anchor='mm')
    d.text((650,1035),'数值为百分点；蓝色为融合方案较高，红色为较低',font=font(25),fill=DARK,anchor='mm')
    return save(im,name),arr

def scalar(df,ratio,protocol,condition,variant):
    z=df[(df.ratio==ratio)&(df.protocol==protocol)&(df.condition==condition)&(df.variant==variant)]
    assert len(z)==1,(ratio,protocol,condition,variant,len(z))
    return z.iloc[0]

def result_rows(df,ratio):
    rows=[]
    for prot,cond,label in CONDITIONS:
        b=scalar(df,ratio,prot,cond,VARIANTS[0]); f=scalar(df,ratio,prot,cond,VARIANTS[1])
        rows.append([label,f'{b.balanced_acc_mean*100:.2f} ± {b.balanced_acc_std*100:.2f}',f'{f.balanced_acc_mean*100:.2f} ± {f.balanced_acc_std*100:.2f}',f'{(f.balanced_acc_mean-b.balanced_acc_mean)*100:+.2f}',f'{b.acc_mean*100:.2f} / {f.acc_mean*100:.2f}',f'{b.f1_macro_mean*100:.2f} / {f.f1_macro_mean*100:.2f}'])
    return rows

def subset_rows(df,ratio):
    rows=[]
    for subset in [4101,4102,4103]:
        for prot,cond,label in [CONDITIONS[0],CONDITIONS[1],CONDITIONS[2]]:
            sub=df[(df.ratio==ratio)&(df.subset==subset)&(df.protocol==prot)&(df.condition==cond)]
            b=sub[sub.variant==VARIANTS[0]].iloc[0]; f=sub[sub.variant==VARIANTS[1]].iloc[0]
            rows.append([str(subset),label,f'{100*b.balanced_acc_mean:.2f} ± {100*b.balanced_acc_std:.2f}',f'{100*f.balanced_acc_mean:.2f} ± {100*f.balanced_acc_std:.2f}',f'{100*(f.balanced_acc_mean-b.balanced_acc_mean):+.2f}'])
    return rows

def main():
    FIG.mkdir(parents=True,exist_ok=True); OUT.parent.mkdir(parents=True,exist_ok=True)
    win=pd.read_csv(WIN/'between_subset_summary.csv'); win0=pd.read_csv(WIN/'mean_std.csv'); winpair=pd.read_csv(WIN/'paired_mean_std.csv')
    small=pd.read_csv(SMALL/'transfer_between_subset_summary.csv'); small0=pd.read_csv(SMALL/'transfer_equal_direction_mean_std.csv'); smallpair=pd.read_csv(SMALL/'paired_mean_std.csv')
    full=pd.read_csv(FULL/'transfer_equal_direction_mean_std.csv'); fullpair=pd.read_csv(FULL/'paired_mean_std.csv')
    # Full budgets occur in separate outputs, while small budgets have subset-level SDs.
    win_all=pd.concat([win,win0[win0.ratio==1.0]],ignore_index=True)
    small_all=pd.concat([small,full],ignore_index=True)
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
    normal=doc.styles['Normal']; normal.font.name='Times New Roman'; normal._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); normal.font.size=Pt(10)
    for name in ['Title','Heading 1','Heading 2']:
        s=doc.styles[name]; s.font.color.rgb=RGBColor(0,0,0); pr=s._element.get_or_add_pPr(); b=pr.find(qn('w:pBdr'))
        if b is not None: pr.remove(b)
    title(doc,'V3完整实验流程图表与逻辑分析')
    para(doc,'本文按系数选择、记录内随机窗口、全样本跨负载、小样本跨负载的顺序整理全部V3实验。各部分先交代实验要回答的问题，再给出完整条件的数值图表，并解释结果如何引出下一组实验。数据来自已保存的正式结果，不重新训练或重新测试。')
    para(doc,'核心发现是，融合加辅助分类方案在低训练预算和跨负载同时存在时显示较强的平均优势。该优势随训练预算增加而缩小，且部分方向和外圈类别出现退化。对照只能评价完整方案，不能把增益分别归因于物理分支、转速、参数量、门控或辅助损失。')
    heading(doc,'一  实验流程与图表阅读方法')
    fig(doc,create_experiment_flow_figure(),'图1  V3数据划分、训练和评价的整体关系。各正式阶段内先完成并锁定模型，再统一完成该阶段新增测试。',6.5)
    caption(doc,'表1  四个V3正式阶段的任务和规模')
    table(doc,['阶段','研究作用','模型与评价规模','与下一阶段的关系'],[['系数选择','确定统一辅助系数','30个新模型，30项干净测试','锁定λ=0.3'],['随机窗口','同记录条件下的训练预算和噪声检验','190新模型加10个引用；2700指标','暴露同记录结果的外推限制'],['全样本跨负载','检查单来源模型直接迁移','80模型，240目标评价，3240指标','建立跨负载全预算参照'],['小样本跨负载','考察低预算与负载变化同时存在','720模型，2160目标评价，29160指标','检验最困难组合与方向差异']])
    para(doc,'纯干净训练只评价干净测试。AWGN混合训练评价干净以及-4、-2、0、+2、+4 dB五档，每档包含5个固定噪声实例。每个模型先平均同档噪声实例。随机窗口和小样本跨负载的10%、20%、50%行先对每个子集内5个训练种子求均值，再对3个子集均值求平均和样本标准差。全样本行的标准差来自5个训练种子。两类标准差含义不同。')
    para(doc,'跨负载整体表先对12个有向迁移任务等权，再汇总重复。方向图保留每个方向及子集。子集来自同一训练池，方向复用来源模型和目标数据，噪声实例不是独立样本。图表中的标准差是描述性波动，不是置信区间。')
    heading(doc,'二  系数选择 建立后续比较的共同设置')
    fig(doc,selection_chart(),'图2  五个辅助损失系数的验证交叉熵。误差线为五个训练种子的样本标准差；系数按预定均值规则选定。',5.8)
    selection=json.loads((SEL/'selection.json').read_text(encoding='utf-8'))['candidates']; sr=[]
    evals=pd.read_csv(SEL/'mean_std.csv')
    for x in selection:
        r=evals[(evals.variant=='data_aux_ce') & (evals.aux_weight==x['aux_weight'])].iloc[0]
        sr.append([f"{x['aux_weight']:.2f}",f"{x['mean_best_validation_ce']:.4g}",f"{x['sample_sd']:.3g}",f'{100*r.balanced_acc_mean:.2f} ± {100*r.balanced_acc_std:.2f}'])
    caption(doc,'表2  五档系数的验证值与单独列示的干净测试BA')
    table(doc,['λ','验证CE均值','五种子SD','干净测试BA（%）'],sr)
    para(doc,'λ=0.3的五种子最佳验证CE均值为5.7021×10⁻⁶，在预定规则下最低，因此固定用于后续实验。其与1.0的均值差为1.10×10⁻⁷，小于种子波动，不能称为普适最优。表中测试BA只作历史结果披露，不参与该系数选择。选参时使用全负载随机窗口验证数据，且本阶段已有测试曝光，这限制了后续跨负载结论的确认性。')
    heading(doc,'三  记录内随机窗口训练预算与噪声实验')
    para(doc,'本组实验回答同一批原始记录切窗后，训练数据减少和人工加噪是否影响两种方法。固定训练、验证和测试池分别为481、103和103窗。10%、20%、50%训练预算对应49、98、242窗，并分别采用3个固定嵌套子集。全样本为481窗。')
    for index,ratio in enumerate(RATIOS):
        heading(doc,f'3.{index+1}  训练预算 {int(ratio*100)}%',2)
        if ratio<1:
            caption(doc,f'表{3+index*2}  随机窗口 {int(ratio*100)}%预算全部测试条件')
        else: caption(doc,'表9  随机窗口全样本预算全部测试条件')
        table(doc,['训练 / 测试','单分支BA ± SD','融合BA ± SD','BA差值','ACC 双方法','Macro-F1 双方法'],result_rows(win_all,ratio),fontsize=8)
        fig(doc,canvas_chart(win_all,ratio,f'window_snr_{int(ratio*100)}.png'),f'图{3+index}  随机窗口{int(ratio*100)}%预算下AWGN训练模型的五档噪声BA。干净条件见相邻表格，未作为信噪比点。',6.1)
        b=scalar(win_all,ratio,'awgn_mixed','white_-4',VARIANTS[0]); f=scalar(win_all,ratio,'awgn_mixed','white_-4',VARIANTS[1])
        para(doc,f'在-4 dB时，数据单分支BA为{100*b.balanced_acc_mean:.2f}%，融合方案为{100*f.balanced_acc_mean:.2f}%，差值为{100*(f.balanced_acc_mean-b.balanced_acc_mean):+.2f}个百分点。图中五档噪声曲线与表中干净条件共同反映绝对性能，不应把平均差值写成每个子集和类别都改善。')
        if ratio<1:
            caption(doc,f'表{4+index*2}  随机窗口 {int(ratio*100)}%三个子集的代表性条件')
            table(doc,['子集','训练 / 测试','单分支BA ± 种子SD','融合BA ± 种子SD','差值'],subset_rows(win0,ratio),fontsize=8)
    para(doc,'随机窗口10%预算的对照模型出现两个退化训练实例，较大的平均差值部分来自融合方案减少优化失败。预算增加至20%以上后多数结果接近饱和，纯干净20%、50%和全样本条件存在融合略低的结果。由于训练与测试共享记录，本组主要说明记录内识别和训练稳定性，不能直接证明跨负载泛化。下一组因此改变划分单位。')
    heading(doc,'四  全样本单源跨负载实验')
    para(doc,'分别用0、1、2、3 HP中的一个来源负载训练及验证，并直接测试其余三个负载，构成12个方向。来源完整训练窗口分别为73、156、160、160，验证窗口分别为19、39、40、40。方向的样本预算和数据组成并不完全相同。')
    caption(doc,'表10  全样本跨负载12方向等权的全部测试条件')
    table(doc,['训练 / 测试','单分支BA ± 种子SD','融合BA ± 种子SD','BA差值','ACC 双方法','Macro-F1 双方法'],result_rows(full,1.0),fontsize=8)
    fig(doc,canvas_chart(full,1.0,'full_snr.png'),'图7  全样本跨负载12方向等权的五档AWGN测试BA。',6.1)
    for k,(prot,cond,label) in enumerate([CONDITIONS[0],CONDITIONS[1],CONDITIONS[2]]):
        path,arr=heat(fullpair,1.0,prot,cond,'balanced_acc_mean',f'full_direction_{k}.png')
        fig(doc,path,f'图{8+k}  全样本跨负载{label}时12方向的配对BA差值。含噪条件先平均同档噪声，再平均五种子。',4.35)
        para(doc,f'{label}的12方向中，融合方案BA差值最低为{arr.min():+.2f}个百分点，最高为{arr.max():+.2f}个百分点。方向结果用于检查整体平均是否掩盖局部退化。')
    para(doc,'AWGN训练、-4 dB条件下，12方向等权BA由84.31%增至89.51%，但2 HP到1 HP方向低0.46个百分点。融合方案从AWGN干净到-4 dB下降4.30个百分点，数据单分支下降2.71个百分点。融合方案绝对BA较高，不表示相对噪声退化更小。')
    heading(doc,'五  小样本单源跨负载实验')
    para(doc,'本组在单一来源负载内进一步限制训练窗口。0 HP来源的10%、20%、50%分别为9、16、37窗；1 HP为17、33、78窗；2和3 HP均为17、34、80窗。验证仍使用19至40个标签窗口，因此训练窗口数不能等同于总标注预算。')
    for index,ratio in enumerate([.1,.2,.5]):
        heading(doc,f'5.{index+1}  来源训练预算 {int(ratio*100)}%',2)
        caption(doc,f'表{11+index*2}  小样本跨负载 {int(ratio*100)}%预算全部测试条件')
        table(doc,['训练 / 测试','单分支BA ± 子集SD','融合BA ± 子集SD','BA差值','ACC 双方法','Macro-F1 双方法'],result_rows(small,ratio),fontsize=8)
        fig(doc,canvas_chart(small,ratio,f'small_snr_{int(ratio*100)}.png'),f'图{11+index*3}  {int(ratio*100)}%来源预算下12方向等权的五档AWGN测试BA。',6.1)
        caption(doc,f'表{12+index*2}  {int(ratio*100)}%来源预算各子集的代表性条件')
        table(doc,['子集','训练 / 测试','单分支BA ± 种子SD','融合BA ± 种子SD','差值'],subset_rows(small0,ratio),fontsize=8)
        path,arr=heat(smallpair,ratio,'awgn_mixed','white_-4','balanced_acc_mean',f'small_ba_{int(ratio*100)}.png',True)
        fig(doc,path,f'图{12+index*3}  {int(ratio*100)}%预算下12方向、三个子集的-4 dB BA配对差值。每格为五种子均值。',4.35)
        para(doc,f'该预算下36个方向与子集组合的BA差值范围为{arr.min():+.2f}至{arr.max():+.2f}个百分点。颜色保留负差，不能把方向或子集当作相互独立的数据集。')
        path,arr_or=heat(smallpair,ratio,'awgn_mixed','white_-4','recall_OR_mean',f'small_or_{int(ratio*100)}.png',True)
        fig(doc,path,f'图{13+index*3}  {int(ratio*100)}%预算下12方向、三个子集的-4 dB外圈召回配对差值。',4.35)
        para(doc,f'外圈召回差值范围为{arr_or.min():+.2f}至{arr_or.max():+.2f}个百分点；其中{int((arr_or<0).sum())}个方向与子集组合为负。该结果要求同时报告类别风险，不能只用整体BA判断。')
    heading(doc,'六  按类别、预算与代价汇总',1)
    rows=[]
    for ratio in [.1,.2,.5,1.0]:
        for variant,name in zip(VARIANTS,NAMES):
            r=scalar(small_all,ratio,'awgn_mixed','white_-4',variant)
            rows.append([f'{int(ratio*100)}%',name]+[f'{100*r[f"recall_{c}_mean"]:.2f}' for c in ['N','IR','OR','B']])
    caption(doc,'表17  不同来源预算下12方向等权类别召回，AWGN训练与-4 dB测试')
    table(doc,['预算','方法','正常N','内圈IR','外圈OR','滚动体B'],rows)
    para(doc,'10%预算时，融合方案的正常和内圈召回分别为99.15%和94.73%，数据单分支分别为24.90%和26.60%。20%预算下融合方案的外圈召回为84.20%，低于数据单分支的85.94%。由此可见，整体BA优势主要来自部分困难类别的改善，并不意味着每一类都改善。')
    cost=pd.read_csv(BASE/'load_small_20261025_2017'/'computational_cost.csv').groupby('variant').agg(parameters=('parameters','mean'),latency=('latency_ms','mean'),memory=('peak_gpu_mib','mean'),epoch=('train_seconds_per_epoch','mean'))
    caption(doc,'表18  小样本跨负载模型成本的实测均值')
    table(doc,['方法','参数量','单样本延迟 ms','峰值显存 MiB','每epoch训练秒数'],[[name,f"{cost.loc[v,'parameters']:.0f}",f"{cost.loc[v,'latency']:.2f}",f"{cost.loc[v,'memory']:.2f}",f"{cost.loc[v,'epoch']:.3f}"] for v,name in zip(VARIANTS,NAMES)])
    para(doc,'单样本延迟包含窗口归一化、在线物理特征和网络计算，批量为1，10次预热后同步重复50次，不含磁盘读取和主机到设备传输。训练为批量32。GPU为NVIDIA GeForce RTX 5060 Laptop GPU，编码器使用BF16，其余为FP32。模型间的实际epoch数、更新数和每轮耗时不完全相同。Conv1d和Linear FLOPs不包括物理FFT，不能用于宣称完整计算量相同。')
    heading(doc,'七  将各组实验连成一条证据线',1)
    para(doc,'系数选择首先固定辅助损失权重，为后续配对比较提供共同设置。记录内随机窗口实验显示，完整融合方案在10%训练预算下减轻训练不稳定，但20%以上接近饱和。由于随机划分共享原始记录，下一步需要检查负载变化。全样本跨负载实验表明，12方向平均BA较高，但方向级结果存在负差。随后小样本跨负载进一步把训练预算压低，发现平均优势明显扩大，并随样本数量增加而缩小。类别和方向图说明这一平均优势主要伴随正常、内圈和滚动体召回改善，外圈召回和部分方向仍不稳定。最后，计算成本说明性能提升伴随参数、延迟与显存开销。')
    para(doc,'这条证据线支持的结论是，当前固定CWRU划分下，融合加辅助分类完整方案在低预算直接跨负载评价中具有经验优势。由于目标负载曾参与选参与历史开发，且训练验证共享记录，它不能证明未知设备上的确认性泛化。当前仅有两组整体对照，也不能确定哪一组件造成优势。最有价值的后续补充是参数量和转速控制、去辅助损失及物理分支消融，并在记录级或轴承实例级隔离的新数据上验证。')
    heading(doc,'附录  结果文件与统计层级',1)
    para(doc,'选参来自 experiments/CWRU/实验结果/CWRU_V3/run_20261022_2105/selection_evaluation；随机窗口来自 window_random_20261023_0853/evaluation；全样本跨负载来自 load_full_20261024_0950/evaluation；小样本跨负载来自 load_small_20261025_2017/evaluation。表格取值为对应 mean_std、between_subset_summary 或 transfer_between_subset_summary；方向差值取 paired_mean_std。每个目录还保存逐种子、逐噪声实例、混淆矩阵和预测。')
    para(doc,'正文对每项预算给出全部测试条件，并在代表性条件下展示子集和方向。以下附表进一步列出每个预算、每个子集、每个负载方向和全部测试条件的配对差值。每格两个数字依次为BA差值和外圈召回差值，均以融合方案减数据单分支，单位为百分点。负值表示融合方案较低。逐种子、逐噪声实例和每一个混淆单元的原始明细仍保留在正式结果文件中。')
    heading(doc,'附表A  随机窗口各预算和子集的完整条件差值',2)
    hdr=['预算 / 子集']+[c[2].replace('AWGN训练 / ','AWGN / ').replace('纯干净训练 / ','纯干净 / ') for c in CONDITIONS]
    rows=[]
    for ratio in [.1,.2,.5]:
        for subset in [4101,4102,4103]:
            row=[f'{int(ratio*100)}% / {subset}']
            for prot,cond,_ in CONDITIONS:
                sub=winpair[(winpair.ratio==ratio)&(winpair.subset==subset)&(winpair.protocol==prot)&(winpair.condition==cond)]
                assert len(sub)==1
                r=sub.iloc[0]; row.append(f"{100*r.balanced_acc_mean:+.1f} / {100*r.recall_OR_mean:+.1f}")
            rows.append(row)
    caption(doc,'附表A1  随机窗口各预算、子集和测试条件的BA / 外圈召回差值')
    table(doc,hdr,rows,fontsize=7.3)
    heading(doc,'附表B  全样本跨负载各方向的完整条件差值',2)
    rows=[]
    for source in range(4):
        for target in range(4):
            if source==target: continue
            row=[f'{source}→{target} HP']
            for prot,cond,_ in CONDITIONS:
                sub=fullpair[(fullpair.source_load_hp==source)&(fullpair.target_load_hp==target)&(fullpair.protocol==prot)&(fullpair.condition==cond)]
                assert len(sub)==1
                r=sub.iloc[0]; row.append(f"{100*r.balanced_acc_mean:+.1f} / {100*r.recall_OR_mean:+.1f}")
            rows.append(row)
    caption(doc,'附表B1  全样本跨负载各方向和测试条件的BA / 外圈召回差值')
    table(doc,hdr,rows,fontsize=7.3)
    heading(doc,'附表C  小样本跨负载各预算、子集和方向的完整条件差值',2)
    for ratio in [.1,.2,.5]:
        for subset in [4101,4102,4103]:
            rows=[]
            for source in range(4):
                for target in range(4):
                    if source==target: continue
                    row=[f'{source}→{target} HP']
                    for prot,cond,_ in CONDITIONS:
                        sub=smallpair[(smallpair.ratio==ratio)&(smallpair.subset==subset)&(smallpair.source_load_hp==source)&(smallpair.target_load_hp==target)&(smallpair.protocol==prot)&(smallpair.condition==cond)]
                        assert len(sub)==1,(ratio,subset,source,target,prot,cond,len(sub))
                        r=sub.iloc[0]; row.append(f"{100*r.balanced_acc_mean:+.1f} / {100*r.recall_OR_mean:+.1f}")
                    rows.append(row)
            caption(doc,f'附表C  {int(ratio*100)}%预算 子集{subset} 12方向的BA / 外圈召回差值')
            table(doc,hdr,rows,fontsize=7.3)
    doc.core_properties.author=''; doc.core_properties.last_modified_by=''; doc.core_properties.title='V3完整实验流程图表与逻辑分析'
    doc.save(OUT); print(OUT); print('figures',len(doc.inline_shapes),'tables',len(doc.tables))

if __name__=='__main__': main()
