from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(14,6))
ax.set_xlim(0,14); ax.set_ylim(0,6); ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',fc=color,ec='#444444',lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(x,y,u,v): ax.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':'#333333','lw':1.2})
box(.25,3.2,1.3,.8,'振动窗口\n24000点','#f3f3f3')
box(.25,1.0,1.3,.8,'转速与\n几何参数','#fff1df')
box(2.1,4.15,1.8,.85,'扩张残差\n数据编码器','#e8f0fa')
box(4.5,4.15,1.8,.85,'均值与标准差\n逐通道相加','#e8f0fa')
box(2.1,1.25,1.8,.85,'物理描述\n212维','#fdeaf1')
box(4.5,1.25,1.8,.85,'训练内标准化\n物理投影','#fdeaf1')
box(7,3.25,1.9,.9,'逐通道门控\ng = sigmoid MLP','#efeaf8')
box(9.5,3.25,1.8,.9,'残差融合\nz = LN(d+g⊙p)','#e8f5eb')
box(12.1,3.25,1.5,.9,'共享分类头\n融合输出','#e8f5eb')
box(10.8,.8,2.2,.9,'训练时数据辅助CE\n权重 λ = 0.3','#fff1df')
arrow(1.55,3.65,2.1,4.55); arrow(1.55,3.4,2.1,1.85); arrow(1.55,1.4,2.1,1.65)
arrow(3.9,4.57,4.5,4.57); arrow(3.9,1.67,4.5,1.67)
arrow(6.3,4.57,7,3.95); arrow(6.3,1.67,7,3.45)
arrow(8.9,3.7,9.5,3.7); arrow(11.3,3.7,12.1,3.7)
arrow(6.3,4.42,9.5,4.0); arrow(6.3,1.67,9.5,3.35)
arrow(6.3,4.38,10.8,1.35)
ax.text(8.0,5.55,'数据与物理信息来自同一振动观测；仅物理描述另使用转速与几何参数',ha='center',fontsize=10,color='#555555')
ax.text(10.65,.35,'推理只使用融合输出；辅助分类复用同一分类头',ha='center',fontsize=10,color='#aa2344')
fig.tight_layout(); fig.savefig(Path(r'D:\Project\MY draft\thesis_document\figures\masters_architecture_correct.png'),dpi=260)
