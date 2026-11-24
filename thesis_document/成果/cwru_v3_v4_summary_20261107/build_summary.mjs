import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const outputDir = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(outputDir, 'summary_data.json'), 'utf8'));
const workbook = Workbook.create();
const names = ['汇总', '全部指标', '子集统计', '方向明细', '种子与来源'];
const sheets = Object.fromEntries(names.map(name => [name, workbook.worksheets.add(name)]));
const font = 'Microsoft YaHei';
const colors = { ink: '#172B43', muted: '#566779', header: '#274562', rule: '#D6DFE9', v3: '#EEF4FB', v4: '#FFF3EC', baseline: '#F2F4F6' };
const metrics = data.metrics;
const labels = Object.fromEntries(data.metrics.map((metric, i) => [metric, data.metric_labels[i]]));
const keyMetrics = ['acc', 'balanced_acc', 'f1_macro', 'recall_B'];
const budgetOrder = [.1, .2, .5, 1];
const methodOrder = ['baseline', 'V3', 'V4'];
const methodLabel = { baseline: '共同数据单分支', V3: 'V3 原四带融合', V4: 'V4 全频四带融合' };
const ordered = rows => [...rows].sort((a, b) => budgetOrder.indexOf(a.budget) - budgetOrder.indexOf(b.budget) || methodOrder.indexOf(a.method) - methodOrder.indexOf(b.method) || (a.subset ?? 0) - (b.subset ?? 0) || (a.seed ?? 0) - (b.seed ?? 0) || (a.source ?? 0) - (b.source ?? 0) || (a.target ?? 0) - (b.target ?? 0));
const col = index => { let text = ''; for (let n = index + 1; n; n = Math.floor((n - 1) / 26)) text = String.fromCharCode(65 + (n - 1) % 26) + text; return text; };
const cellKey = row => `${row.budget}/${row.method}`;

function base(sheet, lastColumn, lastRow, title, context) {
  const used = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  used.format.font = { name: font, size: 10, color: colors.ink };
  used.format.verticalAlignment = 'center';
  used.format.rowHeight = 25;
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 10;
  sheet.getRange('A2').values = [[title]];
  sheet.getRange('A2').format.font = { name: font, size: 15, bold: true, color: colors.ink };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 30;
  sheet.getRange(`A2:${lastColumn}2`).format.borders = { bottom: { color: colors.rule, style: 'thin' } };
  sheet.getRange('A3').values = [[context]];
  sheet.getRange('A3').format.font = { name: font, size: 10, color: colors.muted };
}

function header(sheet, row, titles) {
  const range = sheet.getRangeByIndexes(row - 1, 0, 1, titles.length);
  range.values = [titles];
  range.format.fill = colors.header;
  range.format.font = { name: font, size: 10, bold: true, color: '#FFFFFF' };
  range.format.horizontalAlignment = 'center';
  range.format.rowHeight = 33;
  range.format.wrapText = true;
  range.format.borders = { insideVertical: { color: '#FFFFFF', style: 'thin' } };
}

function methodFill(sheet, row, method, lastColumn) {
  sheet.getRange(`A${row}:${lastColumn}${row}`).format.fill = colors[method === 'baseline' ? 'baseline' : method.toLowerCase()];
}

function metricHeaders(keys, uncertainty = '种子SD') {
  return keys.flatMap(metric => [`${labels[metric]}均值`, uncertainty]);
}

// Five seed observations are the inputs to the main statistics.
const inputs = sheets['种子与来源'];
base(inputs, 'N', 105, '种子层结果与实验来源', '每行已在同一训练种子内等权平均12方向及对应固定子集。数值单位为百分比。来源与口径见 M 列。');
header(inputs, 6, ['训练预算', '模型', '训练种子', ...metrics.map(m => `${labels[m]}（%）`)]);
const seedRows = ordered(data.per_seed);
inputs.getRange(`A7:J${6 + seedRows.length}`).values = seedRows.map(row => [row.budget_label, methodLabel[row.method], row.seed, ...metrics.map(m => row[m])]);
inputs.getRange(`D7:J${6 + seedRows.length}`).setNumberFormat('0.00');
inputs.getRange(`C7:C${6 + seedRows.length}`).setNumberFormat('0');
inputs.getRange('A1:A105').format.columnWidth = 14;
inputs.getRange('B1:B105').format.columnWidth = 25;
inputs.getRange('C1:J105').format.columnWidth = 16;
inputs.getRange('L1:L105').format.columnWidth = 3;
inputs.getRange('M1:M105').format.columnWidth = 20;
inputs.getRange('N1:N105').format.columnWidth = 100;
inputs.freezePanes.freezeRows(6);
const seedGroups = new Map();
seedRows.forEach((row, i) => {
  const key = cellKey(row);
  if (!seedGroups.has(key)) seedGroups.set(key, []);
  seedGroups.get(key).push(i + 7);
});
for (const rows of seedGroups.values()) if (rows.length !== 5 || rows[4] - rows[0] !== 4) throw new Error('每组须恰有5个连续的训练种子');

header(inputs, 71, ['训练预算', '训练种子', ...metrics.map(m => `${labels[m]}配对差`)]);
const pairedRows = [];
const pairGroups = new Map();
for (const budget of budgetOrder) {
  const oldRows = seedGroups.get(`${budget}/V3`);
  const newRows = seedGroups.get(`${budget}/V4`);
  const group = [];
  for (let i = 0; i < 5; i++) {
    const row = 72 + pairedRows.length;
    pairedRows.push([budget === 1 ? '全样本' : `${budget * 100}%`, 1706 + i]);
    inputs.getRange(`A${row}:B${row}`).values = [pairedRows.at(-1)];
    inputs.getRange(`C${row}:I${row}`).formulas = [metrics.map((m, j) => `=${col(j + 3)}${newRows[i]}-${col(j + 3)}${oldRows[i]}`)];
    group.push(row);
  }
  pairGroups.set(budget, group);
}
inputs.getRange('C72:I91').setNumberFormat('+0.00;-0.00;0.00');
inputs.getRange('M6:N6').values = [['实验来源', '本次筛选纯干净协议及干净测试条件']];
inputs.getRange('M6:N6').format.fill = colors.header;
inputs.getRange('M6:N6').format.font = { name: font, size: 10, color: '#FFFFFF', bold: true };
data.sources.forEach((source, i) => {
  const row = 7 + i;
  const budgets = source.budgets.map(b => b === 1 ? '全样本' : `${b * 100}%`).join('、');
  inputs.getRange(`M${row}:N${row}`).values = [[`${source.scheme} ${budgets}`, `${source.source}/evaluation/metrics.csv`]];
});
const notes = [
  ['原方案', 'V3 四带为0.5–3、3–6、6–12、12–20 kHz，辅助损失系数 λ＝0.3。'],
  ['新方案', 'V4 四带为0–3、3–6、6–12、12–24 kHz，辅助损失系数 λ＝1.0。'],
  ['训练条件', '全部为纯干净来源训练、来源验证选模及目标负载测试。每配置5个训练种子，固定种子1706–1710。'],
  ['统一种子口径', '每个训练种子先对12个迁移方向及3个固定子集等权平均，再求5个种子的均值与样本标准差。全样本只有完整训练池。'],
  ['子集间标准差', '子集统计表另列3个子集各自五种子均值之间的样本标准差。它与种子SD描述不同的波动。'],
  ['共同基线', '两版相同预算的600组数据单分支概率、标签及窗口索引一致，本表合并为共同数据单分支。'],
  ['比较范围', '两版同时改变了频带和 λ，结果用于比较两套完整配置。历史测试已曝光，λ 由全负载随机窗口验证选择，频带参考全数据事后分析，因此属于探索性评价。'],
  ['不确定性范围', '5个种子的样本标准差描述固定数据条件下的优化随机性。子集和方向共享记录，不能作为独立数据集重复。'],
  ['验证标签预算', '全负载选参另用103个验证窗口。来源0、1、2、3 HP分别另用19、39、40、40个完整来源验证窗口。'],
  ['复算核验', `保存预测复算${data.verification.metric_rows_recomputed}行指标、${data.verification.confusion_cells_recomputed}个混淆单元，最大绝对误差${data.verification.maximum_metric_absolute_error.toExponential(2)}。`],
];
notes.forEach(([label, text], i) => {
  const row = 15 + i;
  inputs.getRange(`M${row}:N${row}`).values = [[label, text]];
  inputs.getRange(`M${row}`).format.font.bold = true;
  inputs.getRange(`N${row}`).format.wrapText = true;
  inputs.getRange(`M${row}:N${row}`).format.rowHeight = text.length > 90 ? 56 : 40;
});

// All seven means and seed standard deviations are calculated from the seed inputs.
const all = sheets['全部指标'];
base(all, 'P', 29, '七项指标与五种子标准差', '纯干净跨负载。均值为百分比，SD为百分点。每组5个种子，先等权平均方向与子集。');
header(all, 6, ['训练预算', '模型', ...metricHeaders(metrics)]);
const summaryRows = ordered(data.summary);
const summaryMap = new Map();
summaryRows.forEach((item, i) => {
  const row = i + 7;
  const seeds = seedGroups.get(cellKey(item));
  all.getRange(`A${row}:B${row}`).values = [[item.budget_label, methodLabel[item.method]]];
  all.getRange(`C${row}:P${row}`).formulas = [metrics.flatMap((m, j) => {
    const inputColumn = col(j + 3);
    const range = `'种子与来源'!${inputColumn}${seeds[0]}:${inputColumn}${seeds[4]}`;
    return [`=AVERAGE(${range})`, `=STDEV.S(${range})`];
  })];
  summaryMap.set(cellKey(item), row);
  methodFill(all, row, item.method, 'P');
});
header(all, 22, ['训练预算', '比较', ...metrics.flatMap(m => [`${labels[m]}差值`, '配对SD'])]);
data.deltas.forEach((item, i) => {
  const row = i + 23;
  const oldRow = summaryMap.get(`${item.budget}/V3`);
  const newRow = summaryMap.get(`${item.budget}/V4`);
  const pairs = pairGroups.get(item.budget);
  all.getRange(`A${row}:B${row}`).values = [[item.budget_label, 'V4 减 V3（百分点）']];
  all.getRange(`C${row}:P${row}`).formulas = [metrics.flatMap((m, j) => [
    `=${col(j * 2 + 2)}${newRow}-${col(j * 2 + 2)}${oldRow}`,
    `=STDEV.S('种子与来源'!${col(j + 2)}${pairs[0]}:${col(j + 2)}${pairs[4]})`
  ])];
});
all.getRange('A1:A29').format.columnWidth = 13;
all.getRange('B1:B29').format.columnWidth = 28;
all.getRange('C1:P29').format.columnWidth = 15;
all.getRange('C7:P26').setNumberFormat('0.00');

// Compact opening view.
const summary = sheets['汇总'];
base(summary, 'J', 29, '原四带与全频四带实验汇总', '纯干净跨负载。V3 原四带 λ＝0.3，V4 全频四带 λ＝1.0。数值为百分比。');
summary.getRange('A4').values = [['统一五种子SD口径。每个种子先等权平均12方向及固定子集，子集间SD另列。']];
summary.getRange('A4').format.font = { name: font, size: 10, color: colors.muted };
header(summary, 6, ['训练预算', '模型', ...metricHeaders(keyMetrics)]);
summaryRows.forEach((item, i) => {
  const row = i + 7;
  const detailRow = summaryMap.get(cellKey(item));
  summary.getRange(`A${row}:B${row}`).values = [[item.budget_label, methodLabel[item.method]]];
  summary.getRange(`C${row}:J${row}`).formulas = [keyMetrics.flatMap(m => {
    const c = metrics.indexOf(m) * 2 + 2;
    return [`='全部指标'!${col(c)}${detailRow}`, `='全部指标'!${col(c + 1)}${detailRow}`];
  })];
  methodFill(summary, row, item.method, 'J');
});
header(summary, 22, ['训练预算', '比较', ...keyMetrics.flatMap(m => [`${labels[m]}差值`, '配对SD'])]);
data.deltas.forEach((item, i) => {
  const row = i + 23;
  summary.getRange(`A${row}:B${row}`).values = [[item.budget_label, 'V4 减 V3（百分点）']];
  summary.getRange(`C${row}:J${row}`).formulas = [keyMetrics.flatMap(m => {
    const c = metrics.indexOf(m) * 2 + 2;
    return [`='全部指标'!${col(c)}${row}`, `='全部指标'!${col(c + 1)}${row}`];
  })];
});
summary.getRange('A1:A29').format.columnWidth = 13;
summary.getRange('B1:B29').format.columnWidth = 28;
summary.getRange('C1:J29').format.columnWidth = 15;
summary.getRange('C7:J26').setNumberFormat('0.00');
for (const c of ['C', 'E', 'G', 'I']) summary.getRange(`${c}23:${c}26`).setNumberFormat('+0.00;-0.00;0.00');
for (const c of ['C', 'E', 'G', 'I', 'K', 'M', 'O']) all.getRange(`${c}23:${c}26`).setNumberFormat('+0.00;-0.00;0.00');
summary.tabColor = colors.header;

// Fixed-subset statistics preserve the within-subset seed uncertainty.
const subset = sheets['子集统计'];
base(subset, 'Q', 52, '固定子集结果与子集间波动', '上表各子集单列五种子统计。下表将三个子集各自五种子均值作为3个观测计算样本标准差。');
header(subset, 6, ['训练预算', '模型', '固定子集', ...metricHeaders(metrics)]);
const subsetRows = ordered(data.per_subset);
subset.getRange(`A7:Q${6 + subsetRows.length}`).values = subsetRows.map(row => [row.budget_label, methodLabel[row.method], row.subset === 0 ? '完整训练池' : row.subset, ...metrics.flatMap(m => [row[`${m}_mean`], row[`${m}_sd`]])]);
subsetRows.forEach((row, i) => methodFill(subset, i + 7, row.method, 'Q'));
header(subset, 40, ['训练预算', '模型', '子集数量', ...metricHeaders(metrics, '子集间SD')]);
const subsetSdRows = ordered(data.subset_sd);
subset.getRange(`A41:Q${40 + subsetSdRows.length}`).values = subsetSdRows.map(row => [row.budget_label, methodLabel[row.method], 3, ...metrics.flatMap(m => [row[`${m}_mean`], row[`${m}_sd`]])]);
subset.getRange('A1:A52').format.columnWidth = 13;
subset.getRange('B1:B52').format.columnWidth = 26;
subset.getRange('C1:C52').format.columnWidth = 16;
subset.getRange('D1:Q52').format.columnWidth = 16;
subset.getRange('D7:Q49').setNumberFormat('0.00');
subset.freezePanes.freezeRows(6);

const direction = sheets['方向明细'];
base(direction, 'R', 151, '各迁移方向的配对结果', '来源与目标负载单位为HP。小样本每种子先平均3个固定子集，再对5种子计算均值及样本标准差。');
header(direction, 6, ['训练预算', '模型', '来源HP', '目标HP', ...metricHeaders(metrics)]);
const directionalRows = ordered(data.directions);
direction.getRange(`A7:R${6 + directionalRows.length}`).values = directionalRows.map(row => [row.budget_label, methodLabel[row.method], row.source, row.target, ...metrics.flatMap(m => [row[`${m}_mean`], row[`${m}_sd`]])]);
direction.getRange('A1:A151').format.columnWidth = 13;
direction.getRange('B1:B151').format.columnWidth = 26;
direction.getRange('C1:D151').format.columnWidth = 11;
direction.getRange('E1:R151').format.columnWidth = 15;
direction.getRange('E7:R150').setNumberFormat('0.00');
direction.freezePanes.freezeRows(6);
direction.freezePanes.freezeColumns(4);

workbook.recalculate();
const allCalculated = all.getRange('C7:P18').values;
summaryRows.forEach((row, i) => {
  metrics.forEach((metric, j) => {
    for (const [field, offset] of [['mean', 0], ['sd', 1]]) {
      const expected = row[`${metric}_${field}`];
      const actual = allCalculated[i][2 * j + offset];
      if (typeof actual !== 'number' || Math.abs(expected - actual) > 1e-9) throw new Error(`汇总复核不一致 ${cellKey(row)} ${metric}_${field}: ${actual} vs ${expected}`);
    }
  });
});
const deltaCalculated = all.getRange('C23:P26').values;
data.deltas.forEach((row, i) => metrics.forEach((metric, j) => {
  if (Math.abs(deltaCalculated[i][2*j] - row[`${metric}_mean`]) > 1e-9 || Math.abs(deltaCalculated[i][2*j+1] - row[`${metric}_sd`]) > 1e-9) throw new Error(`配对差复核不一致 ${row.budget} ${metric}`);
}));
const checks = await workbook.inspect({ kind: 'table', range: '汇总!A6:J26', include: 'values,formulas', tableMaxRows: 21, tableMaxCols: 10, maxChars: 5000 });
const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 100 }, maxChars: 2000 });
console.log(checks.ndjson);
console.log(errors.ndjson);
for (const [name, range, file] of [
  ['汇总', 'A1:J27', 'summary_preview.png'],
  ['全部指标', 'A1:P19', 'metrics_preview.png'],
  ['子集统计', 'A1:Q13', 'subsets_preview.png'],
  ['方向明细', 'A1:R13', 'directions_preview.png'],
  ['种子与来源', 'A1:J13', 'seeds_preview.png'],
  ['种子与来源', 'M6:N24', 'sources_preview.png'],
]) {
  const preview = await workbook.render({ sheetName: name, range, scale: 1.5, format: 'png' });
  await fs.writeFile(path.join(outputDir, file), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
const destination = path.join(outputDir, 'CWRU_V3_V4_纯干净跨负载对比汇总.xlsx');
await output.save(destination);
await fs.writeFile(path.join(outputDir, 'workbook_verification.json'), JSON.stringify({ summary_values_checked: 12*7*2, paired_delta_values_checked: 4*7*2, data_verification: data.verification, formula_scan: errors.ndjson, workbook: destination }, null, 2));
console.log(destination);
