from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml.ns import qn

root=Path(r'D:\Project\MY draft')
source=root/'thesis_document'/'成果'/'V3全部实验图表与逻辑分析.docx'
out=root/'thesis_document'/'成果'/'V3全部实验图表与逻辑分析_补充训练比例图.docx'
figs=root/'thesis_document'/'成果'/'V3训练比例对比图'
doc=Document(source)
doc.add_page_break()
doc.add_heading('补充 不同训练比例下的单分支与融合模型对比',level=1)

def p(text):
    para=doc.add_paragraph(text); para.paragraph_format.first_line_indent=Pt(20)
    for run in para.runs:
        run.font.name='Times New Roman'; run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体'); run.font.size=Pt(10)

def fig(name,caption):
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.keep_with_next=True
    para.add_run().add_picture(str(figs/name),width=Inches(6.6))
    cap=doc.add_paragraph(caption); cap.paragraph_format.space_after=Pt(5)
    for run in cap.runs:
        run.font.size=Pt(9); run.font.name='Times New Roman'; run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')

p('训练比例使用已完成的10%、20%、50%和100%四档，分别展示记录内随机窗口和单源跨负载结果。每组包含纯干净训练与干净测试、AWGN训练与干净测试、AWGN训练与−4 dB测试。纯干净训练模型没有进行人工加噪测试。')
p('10%、20%、50%的点为三个固定子集各自五种子均值的平均，误差线为三个子集均值的样本标准差。100%没有三个独立子集，点和误差线分别为五训练种子均值及样本标准差。两类误差线口径不同，不应直接将其长度作为稳定性变化的严格证据。')
fig('Random_window_acc.png','补充图1  记录内随机窗口不同训练比例下的准确率。黑色圆点为数据单分支，红色星号为融合加辅助分类方案。')
fig('Random_window_balanced_acc.png','补充图2  记录内随机窗口不同训练比例下的均衡准确率。噪声实例先在每个模型内平均。')
p('随机窗口10%预算时，融合方案在三个条件下的总体表现较高；预算增至20%及以上后两种方法接近饱和，纯干净条件下融合方案并非始终较高。训练与测试窗口共享原始记录，因此这一结果主要反映记录内窗口识别。')
fig('Cross_load_acc.png','补充图3  单源跨负载不同训练比例下的准确率。每个训练种子内先对12个迁移方向等权。')
fig('Cross_load_balanced_acc.png','补充图4  单源跨负载不同训练比例下的均衡准确率。每个训练种子内先对12个迁移方向等权。')
p('AWGN训练与−4 dB测试时，10%预算下单分支和融合方案的12方向等权BA分别为36.17%和85.96%，100%预算下分别为84.31%和89.51%。整体差距随着训练预算增加而缩小。各来源的训练池大小不同，因此相同比例并不代表相同窗口数量。')
p('上述曲线反映方向等权平均值。部分迁移方向及外圈类别仍出现融合方案退化，具体情况应结合正文方向图和附表解释。已有实验没有5%训练比例，故未绘制该数据点。')
doc.save(out)
print(out)
