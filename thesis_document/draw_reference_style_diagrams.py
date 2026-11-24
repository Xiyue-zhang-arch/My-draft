from pathlib import Path
import shutil, math
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.text import Text

ROOT=Path(r'D:\Project\MY draft')
SRC=ROOT/'thesis_document'/'成果'/'V3_论文_公式编号与物理依据_20261031'
OUT=ROOT/'thesis_document'/'成果'/'V3_论文_结构示意图补充_20261031'
OUT.mkdir(parents=True,exist_ok=True)
shutil.copytree(SRC/'media',OUT/'media',dirs_exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42})
blue='#b9d6ee'; violet='#dfd4ea'; green='#e0efd5'; line='#111111'
def setup(w,h,xmax,ymax):
    fig,ax=plt.subplots(figsize=(w,h)); ax.set(xlim=(0,xmax),ylim=(0,ymax)); ax.axis('off');fig.subplots_adjust(0,0,1,1);return fig,ax
def rect(ax,x,y,w,h,label='',fc='white',ec='#aaa',fs=14,weight='normal',lw=.7,ls='-',color='black'):
    p=Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw,linestyle=ls);ax.add_patch(p)
    if label: ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,fontweight=weight,color=color,linespacing=1.1)
    return p
def arr(ax,a,b,lw=1.05,style='-',color=line):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,linewidth=lw,linestyle=style,color=color))
def text(ax,x,y,t,fs=14,weight='normal',ha='center',color='black'):
    ax.text(x,y,t,fontsize=fs,ha=ha,va='center',fontweight=weight,color=color,linespacing=1.05)
def save(fig,name):
    for item in fig.findobj(match=Text):
        item.set_fontsize(item.get_fontsize()*1.35)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/'media'/f'{name}.{ext}',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
def panel(ax,x,y,w,h,title):
    rect(ax,x,y,w,h,ec='#cccccc',lw=.8);text(ax,x+w/2,y+h+.23,title,19,'bold')
def bars(ax,x,y,w,h,colors):
    for i,c in enumerate(colors):rect(ax,x,y+i*h/len(colors),w,h/len(colors)*.93,fc=c,ec='#ccc',lw=.4)

# Reference 1: six white framed panels, in a two-row serpentine layout.
fig,ax=setup(18,9.5,18,9.5)
panels=[(.35,5.15,5.25,3.25,'Data collection'),(6.35,5.15,5.25,3.25,'Data preprocessing'),(12.35,5.15,5.25,3.25,'Physical feature design'),(.35,.65,5.25,3.25,'Fault diagnosis'),(6.35,.65,5.25,3.25,'Feature fusion'),(12.35,.65,5.25,3.25,'Data feature extraction')]
for p in panels:panel(ax,*p)
arr(ax,(5.61,6.77),(6.32,6.77));arr(ax,(11.61,6.77),(12.31,6.77))
arr(ax,(12.34,6.0),(11.83,6.0));arr(ax,(11.83,6.0),(11.83,2.50));arr(ax,(11.83,2.50),(11.56,2.50))
arr(ax,(12.35,2.2),(11.62,2.2));arr(ax,(6.34,2.2),(5.64,2.2))
# Split from preprocessing into the data branch, routed outside panel edges.
ax.plot([9.0,9.0,12.0,12.0,12.35],[5.15,4.55,4.55,3.25,3.25],ls='--',lw=.9,color='black');arr(ax,(12.0,3.25),(12.33,3.25),lw=.9)
for x in [1.35,3.45]:
    ax.add_patch(Circle((x,7.35),.57,fill=False,lw=4.5,ec='#777777'));ax.add_patch(Circle((x,7.35),.30,fill=False,lw=1,ec='#444444'))
    for k in range(8):
        ang=2*math.pi*k/8; ax.add_patch(Circle((x+.43*math.cos(ang),7.35+.43*math.sin(ang)),.065,fc='#ddd',ec='#666',lw=.5))
rect(ax,.83,6.49,4.30,.31,'Rolling bearing vibration',fc='#e9e9e9',ec='none',fs=13,weight='bold')
mat=loadmat(ROOT/'data'/'raw'/'CWRU'/'48k Drive End Bearking Fault Data'/'B007_0.mat')
key=next(k for k in mat if k.endswith('_DE_time')); sig=np.asarray(mat[key]).reshape(-1)[:24000];seg=sig[:2500]; xx=np.linspace(.9,5,seg.size); yy=5.86+.40*seg/(np.max(np.abs(seg))+1e-12)
ax.plot(xx,yy,color='#146cb0',lw=.65);text(ax,3.0,5.38,'48 kHz vibration signal',12,'bold')
rect(ax,6.75,7.20,1.26,.62,'Window\n24k points',fs=13);rect(ax,8.43,7.20,1.30,.62,'Fixed data\nsplits',fs=13);rect(ax,10.05,7.20,1.12,.62,'Train /\nval / test',fs=13)
arr(ax,(8.02,7.51),(8.40,7.51));arr(ax,(9.75,7.51),(10.02,7.51))
rect(ax,6.90,6.39,4.25,.35,'Clean / AWGN mixed training',fc='#e9e9e9',ec='none',fs=13,weight='bold')
text(ax,9.02,5.82,'10%, 20%, 50%, 100% budgets',13);text(ax,9.02,5.49,'record-internal / cross-load',13)
rect(ax,12.73,7.33,1.36,.53,'Vibration',fs=13);rect(ax,14.58,7.33,1.30,.53,'4-band\nenvelope',fs=13);rect(ax,16.22,7.33,.99,.53,'212-D',fs=13)
arr(ax,(14.1,7.59),(14.56,7.59));arr(ax,(15.90,7.59),(16.20,7.59))
text(ax,15.05,6.75,'RPM + geometry  →  fault orders',13);text(ax,15.05,6.28,'harmonics / recurrence / phase',13);text(ax,15.05,5.83,'shape statistics / train-only scaling',13)
rect(ax,12.73,2.67,1.15,.62,'Waveform',fs=13);rect(ax,14.20,2.67,1.24,.62,'Dilated\nCNN',fs=13);rect(ax,15.77,2.67,1.37,.62,'Temporal\nstatistics',fs=13)
arr(ax,(13.9,2.98),(14.18,2.98));arr(ax,(15.46,2.98),(15.74,2.98));text(ax,14.92,1.92,'SE attention + 10 residual layers',13);text(ax,14.92,1.48,'32-D data representation',13)
bars(ax,10.72,1.38,.27,1.27,['#d4e5ce','#c8dabe','#bdd3b3','#aecd9e']);text(ax,9.65,3.22,'Data + physical features',14,'bold');rect(ax,7.05,1.57,1.46,.75,'Layer\nNorm',fs=15);rect(ax,8.89,1.57,1.32,.75,'Gate\n+ residual',fs=14);arr(ax,(10.68,2.0),(10.26,2.0));arr(ax,(8.89,1.95),(8.55,1.95));text(ax,8.5,1.14,'auxiliary loss in training',12)
for i,label in enumerate(['Normal','Inner race','Outer race','Ball']):
    yy=3.08-i*.48;rect(ax,.78,yy-.17,1.25,.34,label,fs=12);arr(ax,(2.26,yy),(2.06,yy))
for j in range(4):ax.add_patch(Circle((3.35,1.43+j*.48),.15,fc='white',ec='#888',lw=.7))
for j in range(3):ax.add_patch(Circle((4.35,1.64+j*.48),.15,fc='white',ec='#888',lw=.7))
for j in range(4):
    for k in range(3):ax.plot([3.50,4.20],[1.43+j*.48,1.64+k*.48],lw=.45,color='#333')
for j in range(4):ax.plot([2.26,3.20],[3.08-j*.48,1.43+(3-j)*.48],lw=.6,color='#333')
text(ax,2.98,1.0,'4-class prediction',12,'bold')
save(fig,'reference1_research_flow')

# Reference 2: pastel blocks and an enlarged dashed SE module.
fig,ax=setup(20,7.5,20,7.5)
blocks=[(.25,5.40,1.40,1.15,'Input\n$1\\times24000$',blue),(2.10,5.40,1.55,1.15,'1×1 Conv\n1 → 32',blue),(4.10,5.40,1.45,1.15,'SE Block',violet),(6.04,5.24,4.35,1.47,'',green),(10.83,5.40,1.55,1.15,'Output\nprojection',blue),(12.83,5.40,1.55,1.15,'Temporal\nAvgPool 16',blue),(14.88,5.40,2.15,1.15,'Mean + Std\nLayerNorm',blue),(17.50,5.40,1.65,1.15,'$d\\in\\mathbb{R}^{32}$',blue)]
for x,y,w,h,l,c in blocks:rect(ax,x,y,w,h,l,c,ec='none',fs=15)
for a,b in [(1.65,2.10),(3.65,4.10),(5.55,6.04),(10.39,10.83),(12.38,12.83),(14.38,14.88),(17.03,17.50)]:arr(ax,(a,5.97),(b,5.97))
text(ax,8.22,6.48,'Dilated residual stack ×10',16)
rect(ax,6.62,5.46,1.05,.56,'Conv1D',fc='#c7d5eb',ec='none',fs=14);rect(ax,7.99,5.46,.95,.56,'tanh',fc='#c7d5eb',ec='none',fs=14);ax.plot([6.04,6.04],[5.97,5.74],color='black',lw=.8);arr(ax,(6.04,5.74),(6.60,5.74));arr(ax,(7.67,5.74),(7.98,5.74));arr(ax,(8.95,5.74),(9.35,5.74));ax.add_patch(Circle((9.55,5.74),.13,fill=False,lw=1));text(ax,9.55,5.74,'+',18);ax.plot([6.4,6.4,9.55,9.55],[5.73,6.12,6.12,5.88],color='black',lw=.8);arr(ax,(9.69,5.74),(10.17,5.74))
ax.plot([4.1,.33],[5.4,4.68],ls='--',color='black',lw=.8);ax.plot([5.55,19.5],[5.4,4.68],ls='--',color='black',lw=.8)
rect(ax,.30,1.05,19.25,3.63,ec='#333',ls='--',lw=.8)
text(ax,1.22,4.15,'$F_0\\in\\mathbb{R}^{32\\times N}$',16)
bars(ax,.65,1.63,1.15,2.07,['#e4a99c','#c8d7e4','#b4d4ec','#ffd0ad'])
positions=[(2.85,2.25,1.70,1.03,'Global\nAvgPool'),(5.45,2.25,1.70,1.03,'Fully\nConnected\n32 → 8'),(8.08,2.25,1.25,1.03,'ReLU'),(10.22,2.25,1.70,1.03,'Fully\nConnected\n8 → 32'),(12.88,2.25,1.45,1.03,'Sigmoid')]
for x,y,w,h,l in positions:rect(ax,x,y,w,h,l,fc=violet,ec='none',fs=14)
for a,b in [(1.80,2.85),(4.55,5.45),(7.15,8.08),(9.33,10.22),(11.92,12.88),(14.33,15.40)]:arr(ax,(a,2.77),(b,2.77))
text(ax,15.18,3.33,'$a\\in(0,1)^{32}$',14);ax.add_patch(Circle((15.65,2.77),.16,fill=False,lw=1));text(ax,15.65,2.77,'×',18)
ax.plot([2.0,2.0,15.65,15.65],[2.72,1.35,1.35,2.60],color='black',lw=.8);arr(ax,(15.83,2.77),(16.75,2.77))
bars(ax,16.92,1.63,1.15,2.07,['#e4a99c','#c8d7e4','#b4d4ec','#ffd0ad']);text(ax,17.48,4.13,'$\\widetilde F_0\\in\\mathbb{R}^{32\\times N}$',16)
save(fig,'reference2_data_branch')

# Reference 3: two outlined streams, feature columns, and a gated residual head.
fig,ax=setup(19,8.6,19,8.6)
rect(ax,.35,4.58,7.35,3.30,ec='#bcbcbc',ls='--');rect(ax,.35,.72,7.35,3.30,ec='#bcbcbc',ls='--')
rect(ax,.72,5.65,.46,1.36,fc='#145ba4',ec='#333');text(ax,.95,5.28,'Vibration\ninput',13,'bold',color='#184c80')
rect(ax,1.55,5.74,1.65,1.15,'Data\nencoder',fs=14);rect(ax,3.62,5.74,1.45,1.15,'Statistics\n+ LayerNorm',fs=13);arr(ax,(1.18,6.31),(1.52,6.31));arr(ax,(3.20,6.31),(3.58,6.31));arr(ax,(5.07,6.31),(5.75,6.31))
rect(ax,5.88,5.12,.62,2.25,ec='#d3d3d3');
for j in range(4):ax.add_patch(Circle((6.19,5.43+j*.51),.20,fc='#2e69b9',ec='none'))
text(ax,6.19,4.85,'Data feature $d$',12,'bold',color='#184c80')
rect(ax,.72,1.79,.46,1.36,fc='#eeeeee',ec='#777');text(ax,.95,1.32,'Vibration\n+ RPM\n+ geometry',12,'bold',color='#184c80')
rect(ax,1.55,1.87,1.65,1.17,'Physics\ndescriptor\n212-D',fs=14);rect(ax,3.62,1.87,1.45,1.17,'Scale +\nMLP',fs=13);arr(ax,(1.18,2.46),(1.52,2.46));arr(ax,(3.20,2.46),(3.58,2.46));arr(ax,(5.07,2.46),(5.75,2.46))
rect(ax,5.88,1.25,.62,2.25,ec='#d3d3d3');
for j in range(4):ax.add_patch(Circle((6.19,1.56+j*.51),.20,fc='#d8d8d8',ec='none'))
text(ax,6.19,.97,'Physical feature $p$',12,'bold',color='#184c80')
rect(ax,7.98,1.15,7.75,6.75,ec='#bcbcbc',ls='--');text(ax,9.4,7.54,'Concatenate',14,'bold',color='#17385d')
rect(ax,8.55,4.58,.58,2.26,ec='#ccc');
for j in range(4):ax.add_patch(Circle((8.84,4.87+j*.50),.19,fc='#2e69b9' if j>=2 else '#d8d8d8',ec='none'))
arr(ax,(6.50,6.23),(8.50,6.23),style='--');arr(ax,(6.50,2.35),(8.50,4.70),style='--')
rect(ax,9.65,5.32,1.35,.85,'Gate MLP\n64→32→32',fs=12);arr(ax,(9.13,5.73),(9.62,5.73));text(ax,9.8,4.95,'Sigmoid',12)
ax.add_patch(Circle((11.62,3.90),.25,fc='white',ec='#333'));text(ax,11.62,3.90,'×',20);arr(ax,(10.33,5.32),(11.48,4.13));ax.plot([6.52,10.92,10.92],[2.35,2.35,3.9],lw=.9,color='black');arr(ax,(10.92,3.9),(11.34,3.9))
ax.add_patch(Circle((12.67,3.90),.25,fc='white',ec='#333'));text(ax,12.67,3.90,'+',20);arr(ax,(11.88,3.90),(12.39,3.90));ax.plot([6.51,7.53,7.53,12.67],[6.78,6.78,7.30,7.30],color='black',lw=.8);arr(ax,(12.67,7.30),(12.67,4.18))
rect(ax,13.18,3.43,1.22,.93,'Layer\nNorm',fs=13);arr(ax,(12.94,3.90),(13.15,3.90));rect(ax,14.75,3.43,.75,.93,'Head',fs=12);arr(ax,(14.4,3.90),(14.72,3.90))
rect(ax,16.08,2.49,2.47,2.82,ec='#c0c0c0',ls='--');
for j,l in enumerate(['Normal','Inner race','Outer race','Ball']):rect(ax,16.29,4.75-j*.53,2.05,.37,l,fs=13)
arr(ax,(15.50,3.9),(16.05,3.9));text(ax,17.30,2.73,'Result',15,'bold',color='#17385d')
ax.plot([6.50,7.31,7.31,14.5,14.5],[5.32,5.32,.50,.50,3.15],ls='--',lw=.8,color='#555');arr(ax,(14.5,3.15),(14.75,3.43),style='--');text(ax,11.0,.75,'Data auxiliary CE: shared head, training only',12,color='#333')
save(fig,'reference3_fusion')

tex=(SRC/'thesis.tex').read_text(encoding='utf-8')
def insert_after(anchor,extra):
    global tex
    if tex.count(anchor)!=1:raise ValueError((anchor,tex.count(anchor)))
    tex=tex.replace(anchor,anchor+'\n\n'+extra,1)
def figure(file,caption):
    return '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{media/'+file+'.pdf}\n\\par\\small '+caption+'\n\\end{figure}\n'
anchor='各小样本预算使用三个固定子集，每个配置采用五个训练种子。'
insert_after(anchor,figure('reference1_research_flow','图1-1 本文研究总体流程')+'图1-1展示数据构造、双分支特征提取、融合及诊断的主要过程。图中的波形取自CWRU振动记录的局部片段，流程图用于说明研究步骤，不表示使用电流信号。')
anchor='数据单分支与完整方案采用同一编码器规格，使二者在原始波形表示路径上具有一致的结构基础。'
insert_after(anchor,figure('reference2_data_branch','图3-4 数据分支及通道注意力结构')+'图3-4给出数据分支的计算过程。残差堆叠由十层膨胀卷积组成；平均池化之后还需计算时间均值与标准差，逐通道相加并归一化。')
anchor='门控网络同时接收d和p，因此g是两类表示共同决定的缩放量。'
insert_after(anchor,figure('reference3_fusion','图3-5 数据分支与物理分支的门控融合结构')+'图3-5中，两路表示先共同生成门控权重。物理表示逐通道乘以门控权重后，与数据表示进行残差相加；虚线所示数据辅助分类仅用于训练，且复用同一个分类头。')
(OUT/'thesis.tex').write_text(tex,encoding='utf-8')
shutil.copy2(SRC/'物理特征文献依据对照.md',OUT/'物理特征文献依据对照.md')
print('Created figures and thesis in',OUT,'waveform key',key)
