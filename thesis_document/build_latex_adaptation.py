from pathlib import Path
import re

root = Path(__file__).resolve().parent / '成果' / 'V3_毕业论文_LaTeX'
s = (root/'converted.tex').read_text(encoding='utf-8')
s = s[s.index('\\subsection{摘 要}'):]
s = re.sub(r'\\subsection\{摘 要\}\\label\{[^}]+\}', r'\\frontmatter\n\\chapter*{中文摘要}\n\\addcontentsline{toc}{chapter}{中文摘要}', s, count=1)
s = re.sub(r'\\subsection\{Abstract\}\\label\{[^}]+\}', r'\\chapter*{Abstract}\n\\addcontentsline{toc}{chapter}{Abstract}', s, count=1)
s = re.sub(r'\\subsection\{目 录\}.*?(?=\\section\{第1章)', lambda m: '\\tableofcontents\n\\mainmatter\n', s, flags=re.S)
s = re.sub(r'\\section\{第[1-6]章\s*([^}]+)\}', lambda m: '\\chapter{' + m.group(1) + '}', s)
s = re.sub(r'\\subsection\{\d+\.\d+\s+', r'\\section{', s)
s = re.sub(r'\\subsubsection\{\d+\.\d+\.\d+\s+', r'\\subsection{', s)
s = s.replace('\\section{辅助损失系数选择与训练过程}', '\\clearpage\n\\section{辅助损失系数选择与训练过程}')
s = s.replace('\\section{参考文献}', '\\chapter*{参考文献}\\addcontentsline{toc}{chapter}{参考文献}')
s = re.sub(r'D:/Project/MY draft/(?:output|thesis_document/成果)/V3_毕业论文_LaTeX/media/', 'media/', s)
equations = [
 r'f_{\mathrm{FTF}}=\frac{f_r}{2}\left(1-\frac{d\cos\theta}{D}\right)',
 r'f_{\mathrm{BPFO}}=\frac{nf_r}{2}\left(1-\frac{d\cos\theta}{D}\right)',
 r'f_{\mathrm{BPFI}}=\frac{nf_r}{2}\left(1+\frac{d\cos\theta}{D}\right)',
 r'f_{\mathrm{BSF}}=\frac{Df_r}{2d}\left[1-\left(\frac{d\cos\theta}{D}\right)^2\right]',
 r'R_h=\log_{10}(P_h/B_h)',
 r'C(f)=\sqrt{\left(\sum_{i=1}^{M}\cos(2\pi ft_i)/M\right)^2+\left(\sum_{i=1}^{M}\sin(2\pi ft_i)/M\right)^2}',
 r'\tilde u_j=\operatorname{clip}\left((u_j-m_j)/\max(\mathrm{IQR}_j,0.05),-8,8\right)',
 r'h_{i+1}=h_i+\tanh\left[C_i(h_i)\right]',
 r'd=\operatorname{LayerNorm}\left[\mu(H)+\sqrt{\operatorname{Var}(H)+10^{-8}}\right]',
 r'g=\operatorname{Sigmoid}\left[W_2\operatorname{ReLU}\left(W_1[d,p]+b_1\right)+b_2\right]',
 r'z=\operatorname{LayerNorm}(d+g\odot p)',
 r'L=\operatorname{CE}[h(z),y]+\lambda\operatorname{CE}[h(d),y]',
 r'J_{\mathrm{val}}=0.5\,\operatorname{CE}_{\mathrm{clean}}+0.1\sum\operatorname{CE}_{\mathrm{SNR}}',
 r'\mathrm{BA}=\frac{1}{4}\sum_k\mathrm{Recall}_k',
]
parts = re.split(r'\\\[.*?\\\]', s, flags=re.S)
assert len(parts)==len(equations)+1, len(parts)
subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₕₙₛᵣ', '0123456789ijkhnsr')
superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')
def clean_text(t):
    t = re.sub(r'([A-Za-z])([₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₕₙₛᵣ]+)', lambda m: '$'+m.group(1)+'_{'+m.group(2).translate(subscripts)+'}$', t)
    t = re.sub(r'(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)', lambda m: '$'+m.group(1)+'^{'+m.group(2).translate(superscripts)+'}$', t)
    for char, cmd in {'θ':'theta','π':'pi','ε':'epsilon','μ':'mu','σ':'sigma','λ':'lambda','⊙':'odot'}.items():
        t=t.replace(char, '\\ensuremath{\\'+cmd+'}')
    t = re.sub(r'\\includegraphics\[width=5\.8in,height=[^]]+\]', r'\\noindent\\includegraphics[width=\\linewidth,keepaspectratio]', t)
    t = t.replace('，F1为', '。F1为').replace('，F1为2', '。F1为2')
    t = t.replace('https://engineering.case.edu/bearingdatacenter/download-data-file.', r'\url{https://engineering.case.edu/bearingdatacenter/download-data-file}.')
    return t
parts = [clean_text(t) for t in parts]
s = ''.join(parts[i]+'\\[\n'+eq+'\n\\]\n' for i,eq in enumerate(equations))+parts[-1]
s = re.sub(r'(?m)^图(\d-\d+ [^\n]+)$', r'\\begin{center}图\1\\end{center}', s)
s = re.sub(r'(?m)^表(\d-\d+ [^\n]+)$', r'\\begin{center}表\1\\end{center}', s)
header = r'''% !TeX program = xelatex
\documentclass[UTF8,12pt,openany,twoside]{ctexbook}
\usepackage[a4paper,top=3cm,bottom=2.5cm,left=2.5cm,right=2.5cm,bindingoffset=1cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc}
\usepackage[hidelinks]{hyperref}
\renewcommand{\baselinestretch}{1.5}
\emergencystretch=3em
\ctexset{chapter={format+=\zihao{-3}\heiti,beforeskip=0pt,afterskip=18pt},section={format+=\zihao{4}\heiti\raggedright},subsection={format+=\zihao{-4}\heiti\raggedright}}
\pagestyle{headings}\setlength{\headheight}{15pt}
\setcounter{tocdepth}{2}
\begin{document}
\begin{titlepage}\centering\vspace*{4cm}{\zihao{1}\heiti 硕士学位论文正文稿\par}\vspace{2cm}{\zihao{2}\heiti 面向小样本与跨负载工况的\\物理引导双流融合轴承故障诊断研究\par}\vspace{2cm}{\zihao{4} 基于 CWRU V3 固定实验\par}\end{titlepage}
'''
(root/'thesis.tex').write_text(header+s+'\n\\end{document}\n', encoding='utf-8')
print('equations',len(equations),'figures',s.count('\\includegraphics'),'tables',s.count('\\begin{longtable}'))
