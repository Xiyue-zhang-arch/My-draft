# V1基线数据使用

全部40个MAT记录，687完整无重叠窗，每窗24000点，48kHz。data只读；各记录尾部不足一窗部分不用，不选择性丢弃完整窗。

| split         | role       |   windows |   N |   IR |   OR |   B |
|:--------------|:-----------|----------:|----:|-----:|-----:|----:|
| window_random | train      |       481 |  49 |  138 |  147 | 147 |
| window_random | validation |       103 |  10 |   29 |   32 |  32 |
| window_random | test       |       103 |  11 |   30 |   31 |  31 |
| load_source_0 | train      |        73 |   8 |   17 |   24 |  24 |
| load_source_0 | validation |        19 |   2 |    5 |    6 |   6 |
| load_source_0 | test_hp1   |       195 |  20 |   55 |   60 |  60 |
| load_source_0 | test_hp2   |       200 |  20 |   60 |   60 |  60 |
| load_source_0 | test_hp3   |       200 |  20 |   60 |   60 |  60 |
| load_source_1 | train      |       156 |  16 |   44 |   48 |  48 |
| load_source_1 | validation |        39 |   4 |   11 |   12 |  12 |
| load_source_1 | test_hp0   |        92 |  10 |   22 |   30 |  30 |
| load_source_1 | test_hp2   |       200 |  20 |   60 |   60 |  60 |
| load_source_1 | test_hp3   |       200 |  20 |   60 |   60 |  60 |
| load_source_2 | train      |       160 |  16 |   48 |   48 |  48 |
| load_source_2 | validation |        40 |   4 |   12 |   12 |  12 |
| load_source_2 | test_hp0   |        92 |  10 |   22 |   30 |  30 |
| load_source_2 | test_hp1   |       195 |  20 |   55 |   60 |  60 |
| load_source_2 | test_hp3   |       200 |  20 |   60 |   60 |  60 |
| load_source_3 | train      |       160 |  16 |   48 |   48 |  48 |
| load_source_3 | validation |        40 |   4 |   12 |   12 |  12 |
| load_source_3 | test_hp0   |        92 |  10 |   22 |   30 |  30 |
| load_source_3 | test_hp1   |       195 |  20 |   55 |   60 |  60 |
| load_source_3 | test_hp2   |       200 |  20 |   60 |   60 |  60 |

逐文件原始路径、通道变量、原始采样点、尾部点和集合用量见[dataset_file_usage.csv](dataset_file_usage.csv)。固定窗口分配见splits下windows.csv及train/validation/test的txt。文本内为全局窗口编号，不是新增数据。窗口随机划分训练测试共享记录，IR014_0两窗不能同时分入三集合；按负载方案来源记录内8:2划分，未抽取共同预算，来源全部训练候选均保留。