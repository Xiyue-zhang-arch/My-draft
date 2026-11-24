function redraw_all(mode, supplementBook)
% Reproduce all comparison figures with native MATLAB graphics.
if nargin == 0, mode = 'all'; end
if nargin < 2, supplementBook = ''; end
assert(any(strcmp(mode, {'all','smoke','mean_confusion','extension','extension_smoke'})));
root = fileparts(mfilename('fullpath'));
if any(strcmp(mode, {'extension','extension_smoke'}))
    stats = load(fullfile(root, 'data_stats.mat'));
    extension = load(fullfile(root, 'data_confusion_extension.mat'));
    assert(isequal(size(extension.matrices),[4 4 2 7 4 2]));
    assert(all(abs(sum(extension.matrices,2)-1)<1e-12,'all'));
    preview = strcmp(mode,'extension_smoke');
    if preview
        destination = fullfile(fileparts(fileparts(fileparts(root))), 'tmp', 'matlab_extension_smoke');
    else
        destination = fullfile(root, 'figures_extension');
    end
    initOutput(destination);
    emitExtension(stats, extension, destination, supplementBook, preview);
    return
end
conf = load(fullfile(root, 'data_confusion.mat'));
assert(all(abs(sum(conf.matrices,2)-1)<1e-12,'all'));
if strcmp(mode, 'mean_confusion')
    emit(meanConfusionFigure(conf), 'confusion_cross_full_clean_mean', ...
        fullfile(root, 'figures'), supplementBook, 480);
    return
end
stats = load(fullfile(root, 'data_stats.mat'));
features = load(fullfile(root, 'data_features.mat'));
assert(isequal(features.sources, conf.sources) && isequal(features.targets, conf.targets));
assert(isequal(size(stats.clean_means), [4 2 7 2]));
assert(isequal(size(stats.awgn_means), [4 2 7 6 2]));
assert(all(isfinite(stats.clean_means(:))) && all(isfinite(stats.awgn_means(:))));

if strcmp(mode, 'smoke')
    destination = fullfile(fileparts(fileparts(fileparts(root))), 'tmp', 'matlab_native_smoke');
else
    destination = fullfile(root, 'figures');
end
initOutput(destination);
book = fullfile(destination, '全部图件.pdf');

if strcmp(mode, 'smoke')
    emit(cleanFigure(stats, 1, 1:3), 'clean_demo', destination, '', 140);
    emit(noiseFigure(stats, 1, 2), 'noise_demo', destination, '', 140);
    emit(confusionFigure(conf, 11), 'confusion_demo', destination, '', 140);
    emit(featureFigure(features, 11), 'feature_demo', destination, '', 140);
    emit(directionFigure(stats, 'clean', 0), 'direction_demo', destination, '', 140);
    emit(lambdaFigure(stats), 'lambda_demo', destination, '', 140);
    return
end

emit(lambdaFigure(stats), '01_lambda_selection', destination, book, 480);
for scene = 1:2
    if scene == 1, code = 'window'; else, code = 'cross'; end
    emit(cleanFigure(stats, scene, 1:3), [code '_clean_overall'], destination, book, 480);
    emit(cleanFigure(stats, scene, 4:7), [code '_clean_recall'], destination, book, 480);
    tags = {'acc','ba','macro_f1','normal','inner_race','outer_race','rolling_element'};
    for metric = [2 1 3:7]
        emit(noiseFigure(stats, scene, metric), [code '_awgn_' tags{metric}], destination, book, 480);
    end
end
emitDirections(stats,destination,book);
for index = 1:13
    key = targetKey(features.sources(index), features.targets(index));
    emit(featureFigure(features, index), ['tsne_' key], destination, book, 480);
end
for index = 1:13
    key = targetKey(conf.sources(index), conf.targets(index));
    emit(confusionFigure(conf, index), ['confusion_' key], destination, book, 480);
end
emit(meanConfusionFigure(conf), 'confusion_cross_full_clean_mean', destination, book, 480);
fprintf('COMPLETE: %d FIG, %d PNG\n', numel(dir(fullfile(destination,'fig','*.fig'))), ...
    numel(dir(fullfile(destination,'png','*.png'))));
end

function initOutput(destination)
assert(~isfolder(destination), '图件输出目录已存在，请先核对现有文件。');
mkdir(destination);
mkdir(fullfile(destination, 'png'));
mkdir(fullfile(destination, 'fig'));
end

function emitDirections(stats,destination,book)
emit(directionFigure(stats,'clean',0), 'cross_clean_direction_ba', destination, book, 480);
codes = budgetCodes();
for budget = 1:4
    emit(directionFigure(stats,'noise',budget), ['cross_awgn_direction_ba_' codes{budget}], destination, book, 480);
end
emit(directionFigure(stats,'rolling',0), 'cross_minus4_direction_rolling_recall', destination, book, 480);
end

function emitExtension(stats,data,destination,book,preview)
if preview
    emit(directionFigure(stats,'clean',0), 'direction_clean', destination, book, 150);
    emit(directionFigure(stats,'noise',4), 'direction_awgn_full', destination, book, 150);
    emit(extendedConfusionFigure(data,3,1,2), 'confusion_cross_10_minus4', destination, book, 150);
    return
end
emitDirections(stats,destination,book);
codes = budgetCodes();
conditions = {'pure_clean','mixed_clean','mixed_minus4','mixed_minus2','mixed_0','mixed_plus2','mixed_plus4'};
scenes = {'window','cross_mean'};
for scene = 1:2
    for condition = 1:7
        for budget = 1:4
            name = strjoin({'confusion',scenes{scene},codes{budget},conditions{condition}},'_');
            emit(extendedConfusionFigure(data,condition,budget,scene), name, destination, book, 480);
        end
    end
end
end

function result = cleanFigure(data, scene, metrics)
if numel(metrics)==3, dims = [1 3]; sizePx = [1300 445]; else, dims = [2 2]; sizePx = [1120 790]; end
result = baseFigure(sizePx);
layout = makeLayout(result,dims(1),dims(2));
title(layout,[sceneName(scene) ' · 纯干净训练与测试'],'FontName','SimSun','Color','k','FontSize',19);
interval = barLimits(data.clean_means(:,:,metrics,scene), data.clean_sds(:,:,metrics,scene));
for i = 1:numel(metrics)
    m = metrics(i);
    ax = nexttile(layout);
    bars = barPanel(ax, data.clean_means(:,:,m,scene), data.clean_sds(:,:,m,scene), interval);
    title(ax, metricName(m), 'FontName','SimSun','Color','k');
    if i==1, comparisonLegend(layout, ax, bars); end
end
footer(result,'柱高为五个训练种子的均值。误差线为样本标准差。小样本先等权平均固定子集。');
end

function result = noiseFigure(data, scene, metric)
result = baseFigure([1430 815]);
layout = makeLayout(result,2,3);
title(layout,[sceneName(scene) ' · AWGN 混合训练 · ' metricName(metric)], ...
    'FontName','SimSun','Color','k','FontSize',19);
conditions = {'-4 dB','-2 dB','0 dB','+2 dB','+4 dB','干净'};
interval = barLimits(data.awgn_means(:,:,metric,:,scene), data.awgn_sds(:,:,metric,:,scene));
for index = 1:6
    ax = nexttile(layout);
    bars = barPanel(ax, data.awgn_means(:,:,metric,index,scene), data.awgn_sds(:,:,metric,index,scene), interval);
    title(ax, conditions{index}, 'FontName','SimSun','Color','k');
    if index==1, comparisonLegend(layout, ax, bars); end
end
footer(result,'误差线为五种子样本标准差。先平均同 SNR 的五次加噪，再等权汇总固定子集；跨负载等权汇总方向。');
end

function h = barPanel(ax, means, sd, interval)
means = reshape(means,4,2);
sd = reshape(sd,4,2);
h = bar(ax, 1:4, means, 1, 'grouped');
h(1).FaceColor = [32 121 180]/255;
h(2).FaceColor = [250 130 22]/255;
hold(ax,'on');
for i = 1:2
    h(i).EdgeColor = 'none';
    errorbar(ax, h(i).XEndPoints, means(:,i), sd(:,i), 'k', ...
        'LineStyle','none','LineWidth',1.05,'CapSize',6,'HandleVisibility','off');
end
styleAxis(ax);
ax.XTick = 1:4;
ax.XTickLabel = budgets();
ax.FontName = 'SimSun';
ax.YLim = interval;
ax.YTick = 20*ceil(interval(1)/20):20:20*floor(interval(2)/20);
xlim(ax,[.43 4.57]);
ylabel(ax,'百分比（%）','FontName','SimSun','Color','k');
xlabel(ax,'训练预算','FontName','SimSun','Color','k');
end

function interval = barLimits(means,sd)
interval = [min(0,5*floor(min(means(:)-sd(:))/5)), max(105,5*ceil((max(means(:)+sd(:))+2)/5))];
end

function comparisonLegend(layout, ax, bars)
leg = legend(ax, bars, {'融合','数据单分支'}, 'Orientation','horizontal','Box','off', ...
    'FontName','SimSun','TextColor','k','FontSize',12);
leg.Layout.Tile = 'north';
end

function result = lambdaFigure(data)
result = baseFigure([1010 640]);
ax = axes(result,'Position',[.13 .195 .81 .65]);
x = 1:numel(data.lambda_values);
means = data.lambda_means(:)*1e6;
sd = data.lambda_sds(:)*1e6;
bars = bar(ax,x,means,.60,'FaceColor','flat','EdgeColor','none');
bars.CData = repmat([32 121 180]/255,numel(x),1);
selected = find(abs(data.lambda_values - data.lambda_constrained)<1e-12);
bars.CData(selected,:) = [250 130 22]/255;
hold(ax,'on');
errorbar(ax,x,means,sd,'k','LineStyle','none','CapSize',7,'LineWidth',1.2);
styleAxis(ax);
ax.XTick = x;
ax.XTickLabel = compose('%.2f',data.lambda_values);
ax.YLim = [0 ceil(max(means+sd)+2)];
xlabel(ax,'辅助损失系数 \lambda','FontName','SimSun','Color','k');
ylabel(ax,'最佳验证交叉熵（\times 10^{-6}）','FontName','SimSun','Color','k');
title(ax,'辅助损失系数选择','FontName','SimSun','Color','k','FontSize',18);
text(ax,selected,means(selected)+sd(selected)+.9,'正式实验 \lambda = 0.3', ...
    'HorizontalAlignment','center','FontName','SimSun','Color','k','FontSize',11);
footer(result,'柱高为五种子均值，误差线为样本标准差。原网格最低均值对应 1.0，橙柱是确认的正式系数。');
end

function result = directionFigure(data, mode, budget)
result = baseFigure([1320 790]);
layout = makeLayout(result,1,2);
anchors = [248 252 255; 225 239 250; 185 216 238; 122 180 221; 53 129 190]/255;
map = interp1(0:25:100,anchors,linspace(0,100,256));
directions = cellstr(compose('%d 到 %d',data.directions(:,1),data.directions(:,2)));
xLabel = '训练预算';
switch mode
    case 'clean'
        name = '纯干净跨负载 · 各方向平衡准确率';
        headings = budgets();
        values = @(m) reshape(data.clean_dir_means(:,m,2,:),4,12).';
    case 'noise'
        b = budgets();
        name = ['AWGN 跨负载 · 各方向平衡准确率 · ' b{budget}];
        headings = {'-4','-2','0','+2','+4','干净'};
        xLabel = '测试条件（SNR，dB）';
        values = @(m) reshape(data.awgn_dir_means(budget,m,2,:,:),6,12).';
    case 'rolling'
        name = 'AWGN 跨负载 · -4 dB 滚动体召回';
        headings = budgets();
        values = @(m) reshape(data.awgn_dir_means(:,m,7,1,:),4,12).';
end
title(layout,name,'FontName','SimSun','Color','k','FontSize',19);
names = {'融合','数据单分支'};
for method=1:2
    ax = nexttile(layout);
    matrix = values(method);
    imagesc(ax,matrix,[0 100]);
    colormap(ax,map);
    styleAxis(ax);
    ax.FontName = 'SimSun';
    ax.TickLength = [0 0];
    ax.XTick=1:numel(headings); ax.XTickLabel=headings;
    ax.YTick=1:12; ax.YTickLabel=directions;
    title(ax,names{method},'FontName','SimSun','Color','k','FontSize',14);
    xlabel(ax,xLabel,'FontName','SimSun','Color','k');
    if method==1
        ylabel(ax,'来源负载到目标负载（HP）','FontName','SimSun','Color','k');
    else
        ax.YTickLabel = [];
    end
    hold(ax,'on');
    for k = 1.5:(size(matrix,2)-.5)
        line(ax,[k k],[.5 12.5],'Color','w','LineWidth',.5);
    end
    for k = 1.5:11.5
        width = .5 + .7*(mod(k-.5,3)==0);
        line(ax,[.5 size(matrix,2)+.5],[k k],'Color','w','LineWidth',width);
    end
    for i=1:size(matrix,1)
        for j=1:size(matrix,2)
            text(ax,j,i,sprintf('%.1f',matrix(i,j)),'HorizontalAlignment','center', ...
                'Color','k','FontName','Times New Roman','FontSize',13);
        end
    end
end
scale = colorbar(ax,'Ticks',0:20:100,'Color','k','FontName','Times New Roman');
scale.Layout.Tile = 'east';
scale.Label.String = '%';
footer(result,'两组共用 0 至 100% 色标。各格为该迁移方向的平均值。');
end

function result = featureFigure(data, index)
old = double(data.X_old{index});
new = double(data.X_new{index});
y = double(data.y{index}(:));
points = {project(old), project(new)};
coordinates = vertcat(points{:});
lower = min(coordinates,[],1); upper = max(coordinates,[],1);
mid = (lower + upper)/2; half = .57 * max(upper-lower);
labels = {'正常','内圈','外圈','滚动体'};
colors = [0 119 179;255 125 0;159 99 189;0 160 15]/255;
names = {'数据单分支','融合'};
result = baseFigure([1190 630]);
layout = makeLayout(result,1,2);
title(layout,[targetTitle(data.sources(index),data.targets(index)) ' · 特征 t-SNE'], ...
    'FontName','SimSun','Color','k','FontSize',19);
for method = 1:2
    ax = nexttile(layout);
    hold(ax,'on');
    h = gobjects(1,4);
    for c=1:4
        xy = points{method}(y==c-1,:);
        h(c) = scatter(ax,xy(:,1),xy(:,2),11,colors(c,:),'filled','MarkerEdgeColor','none');
    end
    styleAxis(ax);
    axis(ax,'square');
    xlim(ax,mid(1)+[-half half]); ylim(ax,mid(2)+[-half half]);
    xlabel(ax,'t-SNE dimension 1','Color','k');
    ylabel(ax,'t-SNE dimension 2','Color','k');
    title(ax,names{method},'FontName','SimSun','Color','k');
    if method==1
        leg=legend(ax,h,labels,'Orientation','horizontal','FontName','SimSun','TextColor','k','Box','off');
        leg.Layout.Tile='north';
    end
end
footer(result,sprintf('同一批 %d 个测试窗，训练种子 1706。两面板使用相同参数分别降维。',length(y)));
end

function xy = project(features)
rng(1706,'twister');
[U,S,~] = svd(features-mean(features,1),'econ');
start = U(:,1:2)*S(1:2,1:2);
start = start * (1e-4 / max(std(start(:,1)),eps));
xy = tsne(features,'Algorithm','exact','NumDimensions',2,'Perplexity',15, ...
    'LearnRate',30,'Exaggeration',4,'Standardize',false,'InitialY',start,'Verbose',0, ...
    'Options',statset('MaxIter',1500));
assert(all(isfinite(xy),'all'));
end

function result = meanConfusionFigure(data)
index = find(data.sources >= 0);
directions = sub2ind([4 4], data.sources(index)+1, data.targets(index)+1);
assert(isequal(sort(directions(:)), find(~eye(4))), '跨负载方向应完整且不重复。');
result = confusionFigure(data,index);
end

function result = confusionFigure(data,index)
if isscalar(index)
    heading = targetTitle(data.sources(index),data.targets(index));
    note = sprintf('按真实类别归一化，五种子等权平均。%d 个测试窗。',data.testWindows(index));
else
    heading = '跨负载全样本纯干净 · 12 个方向平均';
    note = '按真实类别归一化，五个种子和 12 个迁移方向分别等权平均。';
end
result = drawConfusionPair(mean(data.matrices(:,:,:,index),4), ...
    mean(data.ACC(:,index),2), mean(data.BA(:,index),2), data.cmap, heading, note);
end

function result = extendedConfusionFigure(data,condition,budget,scene)
levels = {'干净测试','干净测试','-4 dB','-2 dB','0 dB','+2 dB','+4 dB'};
budgetNames = budgets();
if condition == 1
    protocol = '纯干净训练';
else
    protocol = 'AWGN 混合训练';
end
sceneText = sceneName(scene);
if scene == 2, sceneText = '跨负载 12 方向平均'; end
heading = strjoin({sceneText,budgetNames{budget},protocol,levels{condition}}, ' · ');
weights = {'五种子'};
if budget ~= 4, weights{end+1} = '三子集'; end
if condition >= 3, weights{end+1} = '五次加噪'; end
if scene == 2, weights{end+1} = '12 个方向'; end
note = ['按真实类别归一化。' strjoin(weights,'、') '分别等权平均。'];
result = drawConfusionPair(data.matrices(:,:,:,condition,budget,scene), ...
    data.acc(:,condition,budget,scene), data.ba(:,condition,budget,scene), data.cmap, heading, note);
end

function result = drawConfusionPair(matrices,acc,ba,map,heading,note)
result = baseFigure([1190 620]);
layout = makeLayout(result,1,2);
title(layout,[heading ' · 归一化混淆矩阵'], ...
    'FontName','SimSun','Color','k','FontSize',19);
names = {'数据单分支','融合'};
for method=1:2
    ax = nexttile(layout);
    matrix = matrices(:,:,method);
    imagesc(ax,matrix,[0 1]);
    colormap(ax,map);
    axis(ax,'image');
    styleAxis(ax);
    ax.XTick=1:4; ax.YTick=1:4;
    ax.XTickLabel={'N','IR','OR','B'}; ax.YTickLabel=ax.XTickLabel;
    title(ax,{names{method}, sprintf('ACC = %.2f%%   |   BA = %.2f%%', ...
        acc(method),ba(method))}, ...
        'FontName','SimSun','Color','k');
    xlabel(ax,'预测类别','FontName','SimSun','Color','k');
    ylabel(ax,'真实类别','FontName','SimSun','Color','k');
    hold(ax,'on');
    for k = 1.5:3.5
        line(ax,[.5 4.5],[k k],'Color','w','LineWidth',.6);
        line(ax,[k k],[.5 4.5],'Color','w','LineWidth',.6);
    end
    for i=1:4
        for j=1:4
            value = matrix(i,j);
            valueLabel = text(ax,j,i,sprintf('%.3f',value),'HorizontalAlignment','center', ...
                'Color','k','FontName','Times New Roman','FontSize',15);
            if value>.53, set(valueLabel,'BackgroundColor','w','Margin',1.4); end
        end
    end
end
scale = colorbar(ax,'Ticks',0:.25:1,'TickLabels',compose('%.2f',0:.25:1),'Color','k','FontName','Times New Roman');
scale.Layout.Tile = 'east';
footer(result,[note 'N 正常，IR 内圈，OR 外圈，B 滚动体。']);
end

function value=targetKey(source,target)
if source<0, value='window'; else, value=sprintf('%dhp_to_%dhp',source,target); end
end

function value=targetTitle(source,target)
if source<0, value='窗口划分全样本纯干净'; else, value=sprintf('跨负载全样本纯干净 · %d HP 到 %d HP',source,target); end
end

function value=metricName(index)
labels={'准确率（ACC）','平衡准确率（BA）','宏平均 F1','正常召回','内圈召回','外圈召回','滚动体召回'};
value=labels{index};
end

function value=sceneName(index)
labels={'窗口划分','跨负载'};
value=labels{index};
end

function value=budgets()
value={'10%','20%','50%','全样本'};
end

function value=budgetCodes()
value={'10','20','50','full'};
end

function result=baseFigure(sizePx)
result=figure('Visible','off','Color','w','Units','pixels','Position',[60 60 sizePx], ...
    'DefaultAxesFontName','Times New Roman','DefaultAxesFontSize',11,'DefaultTextColor','k', ...
    'DefaultAxesXColor','k','DefaultAxesYColor','k','DefaultAxesZColor','k');
end

function styleAxis(ax)
set(ax,'Color','w','XColor','k','YColor','k','ZColor','k','LineWidth',.8,'Box','on','TickDir','out', ...
    'Layer','top','FontSize',11);
end

function layout=makeLayout(fig,rows,columns)
layout=tiledlayout(fig,rows,columns,'TileSpacing','compact','Padding','loose');
layout.OuterPosition = [.025 .072 .95 .90];
end

function footer(fig, note)
annotation(fig,'textbox',[.04 .004 .92 .034],'String',note,'EdgeColor','none','Color','k', ...
    'FontName','SimSun','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',9,'FitBoxToText','off');
end

function emit(fig,name,destination,book,dpi)
guard = onCleanup(@() close(fig));
set(findall(fig,'Type','text'),'Color','k');
set(findall(fig,'Type','legend'),'TextColor','k');
drawnow;
savefig(fig,fullfile(destination,'fig',[name '.fig']),'compact');
exportgraphics(fig,fullfile(destination,'png',[name '.png']),'Resolution',dpi,'BackgroundColor','white');
if ~isempty(book)
    exportgraphics(fig,book,'ContentType','vector','BackgroundColor','white','Append',isfile(book));
end
fprintf('EXPORTED %s\n',name);
end
