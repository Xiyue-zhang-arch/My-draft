function plot_results(root)
% Native MATLAB figures. Only black text; method labels omit version names.
root=string(root);
V=jsondecode(fileread(fullfile(root,'report','verification.json')));
assert(V.passed);
T=readtable(fullfile(root,'report','summary.csv'),'TextType','string','VariableNamingRule','preserve');
C=readtable(fullfile(root,'report','mean_normalized_confusion.csv'),'TextType','string','VariableNamingRule','preserve');
out=fullfile(root,'figures');
if ~isfolder(out), mkdir(out); end
set(groot,'DefaultTextColor','k','DefaultAxesXColor','k','DefaultAxesYColor','k', ...
    'DefaultAxesZColor','k','DefaultAxesFontName','Times New Roman', ...
    'DefaultTextFontName','Microsoft YaHei','DefaultAxesFontSize',12, ...
    'DefaultTextInterpreter','none','DefaultLegendTextColor','k');
scenes=["window","load"];
protocols=["clean_only","awgn_mixed"];
conditions=["clean","white_-4","white_-2","white_+0","white_+2","white_+4"];
conditionsText=["干净","−4 dB","−2 dB","0 dB","+2 dB","+4 dB"];
methods=["data","fusion"];
methodText=["数据单分支","融合"];
metrics=["acc","balanced_acc","f1_macro"];
metricText=["ACC (%)","BA (%)","Macro-F1 (%)"];
sceneText=["记录内窗口划分","跨负载12方向平均"];
protocolText=["纯干净训练","AWGN混合训练"];
colors=[0 .447 .741;.85 .325 .098];
labels={'N','IR','OR','B'};
for a=1:2
    for b=1:2
        f=figure('Visible','off','Color','w','Units','pixels','Position',[50 50 1600 530]);
        layout=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
        for metric=1:3
            ax=nexttile(layout);
            means=zeros(6,2); deviations=zeros(6,2);
            for i=1:6
                for method=1:2
                    row=T(T.scene==scenes(a) & T.protocol==protocols(b) & T.method==methods(method) & T.condition==conditions(i),:);
                    assert(height(row)==1);
                    means(i,method)=100*row.(metrics(metric)+"_mean");
                    deviations(i,method)=100*row.(metrics(metric)+"_std");
                end
            end
            bars=bar(ax,means,.82,'grouped'); hold(ax,'on');
            for method=1:2
                bars(method).FaceColor=colors(method,:);
                bars(method).EdgeColor='k';
                errorbar(ax,bars(method).XEndPoints,means(:,method),deviations(:,method), ...
                    'k','LineStyle','none','LineWidth',1,'CapSize',5,'HandleVisibility','off');
            end
            xticks(ax,1:6); xticklabels(ax,conditionsText);
            ylabel(ax,metricText(metric),'Color','k');
            ylim(ax,[0 max(105,max(means+deviations,[],'all')+3)]);
            grid(ax,'on'); ax.GridAlpha=.12; box(ax,'on');
            if metric==1
                legend(ax,bars,methodText,'Location','southoutside','Orientation','horizontal','TextColor','k','FontName','Microsoft YaHei');
            end
        end
        title(layout,sceneText(a)+" · 10% · "+protocolText(b)+"（误差棒为五种子标准差）", ...
            'Color','k','FontName','Microsoft YaHei','FontSize',15);
        name=scenes(a)+"_"+protocols(b)+"_metrics";
        savefig(f,fullfile(out,name+".fig"));
        exportgraphics(f,fullfile(out,name+".png"),'Resolution',600);
        exportgraphics(f,fullfile(out,name+".pdf"),'ContentType','vector'); close(f);
        for i=1:6
            f=figure('Visible','off','Color','w','Units','pixels','Position',[50 50 1250 620]);
            layout=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
            for method=1:2
                ax=nexttile(layout);
                rows=C(C.scene==scenes(a) & C.protocol==protocols(b) & C.method==methods(method) & C.condition==conditions(i),:);
                matrix=zeros(4);
                for k=1:height(rows)
                    matrix(rows.true_label(k)+1,rows.predicted_label(k)+1)=rows.row_fraction(k);
                end
                imagesc(ax,matrix,[0 1]); axis(ax,'square');
                % MATLAB's built-in sequential map; black labels remain readable
                % on a white rectangular backing over dark cells.
                colormap(ax,sky(256)); colorbar(ax,'Color','k','Ticks',0:.25:1);
                xticks(ax,1:4); yticks(ax,1:4); xticklabels(ax,labels); yticklabels(ax,labels);
                xlabel(ax,'预测类别','FontName','Microsoft YaHei','Color','k');
                ylabel(ax,'真实类别','FontName','Microsoft YaHei','Color','k');
                title(ax,methodText(method),'FontName','Microsoft YaHei','Color','k');
                for row=1:4
                    for col=1:4
                        text(ax,col,row,sprintf('%.3f',matrix(row,col)),'HorizontalAlignment','center', ...
                            'Color','k','FontSize',14,'FontName','Times New Roman','BackgroundColor','w','Margin',1);
                    end
                end
            end
            title(layout,sceneText(a)+" · 10% · "+protocolText(b)+" · "+conditionsText(i), ...
                'Color','k','FontName','Microsoft YaHei','FontSize',16);
            name=scenes(a)+"_"+protocols(b)+"_"+conditions(i)+"_confusion";
            savefig(f,fullfile(out,name+".fig"));
            exportgraphics(f,fullfile(out,name+".png"),'Resolution',600);
            exportgraphics(f,fullfile(out,name+".pdf"),'ContentType','image','Resolution',600); close(f);
        end
    end
end
fprintf('MATLAB_FIGURES_COMPLETE: 4 metric figures and 24 paired normalized confusion matrices.\n');
end
