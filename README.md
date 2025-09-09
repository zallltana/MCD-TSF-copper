# MCD-TSF: Multi-modal Conditional Diffusion for Time Series Forecasting

基于扩散模型的多模态时间序列预测框架，专门用于铜价预测，结合数值数据和文本研报信息。

## 项目概述

本项目实现了MCD-TSF（Multi-modal Conditional Diffusion for Time Series Forecasting）模型，用于预测铜价走势。模型结合了：
- **数值数据**：日频的铜价、成交量、持仓等金融指标
- **文本数据**：月频的铜市场研报和分析师观点
- **扩散模型**：生成多个预测样本，提供不确定性量化

## 主要特性

### 🎯 多模态融合
- 数值时间序列数据（日频）
- 文本研报数据（月频）
- 自动时间对齐机制

### 📊 不确定性量化
- 生成多个预测样本（默认15个）
- 提供50%和90%置信区间
- 概率分布分析

### 🔄 重叠预测处理
- 自动处理滑动窗口预测中的重叠问题
- 多种最终预测选择策略：
  - 最新预测（Latest）
  - 平均值（Average）
  - 加权平均（Weighted）
  - 中位数（Median）
  - 集成方法（Ensemble）

### 📈 灵活的目标变量
- 支持二分类预测（涨跌）
- 支持连续值预测（收益率）
- 自动检测目标类型并应用相应指标

## 项目结构

```
MCD-TSF-copper/
├── config/                    # 配置文件
│   └── economy_36_6.yaml     # 模型配置
├── data_provider/            # 数据加载器
│   ├── data_factory.py       # 数据工厂
│   └── data_loader.py        # 数据加载逻辑
├── utils/                    # 工具函数
│   ├── prepare4llm.py        # LLM准备函数
│   ├── utils.py              # 训练和评估工具
│   └── timefeatures.py       # 时间特征
├── main_model.py             # 主模型定义
├── diff_models.py            # 扩散模型
├── exe_forecasting.py        # 执行脚本
├── dataset_forecasting.py    # 数据集定义
├── view_predictions.py       # 基础预测查看器
├── advanced_prediction_viewer.py  # 高级预测分析器
├── overlapping_prediction_handler.py  # 重叠预测处理器
└── select_final_predictions.py       # 最终预测选择器
```

## 安装要求

```bash
# 基础依赖
torch>=2.4.0
torchvision
torchaudio
transformers>=4.20.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm

# CUDA支持（可选）
# 确保安装支持CUDA的PyTorch版本
```

## 数据格式要求

### 数值数据 (copper/numerical/copper/copper.csv)
```csv
date,start_date,end_date,OT,feature1,feature2,...
2010/6/22,2010/6/22,2010/6/22,1,value1,value2,...
```

### 文本数据 (copper/textual/copper/)
- `copper_report.csv`: 研报数据
- `copper_search.csv`: 搜索数据

```csv
start_date,end_date,fact,pred
2010-05-23,2010-06-22,"研报内容...","预测内容..."
```

## 使用方法

### 1. 基础训练和预测

```bash
# 激活环境
conda activate dva38

# 运行训练和预测
python -u exe_forecasting.py \
  --root_path ../copper \
  --data_path copper/copper.csv \
  --config economy_36_6.yaml \
  --seq_len 36 \
  --pred_len 6 \
  --text_len 36 \
  --freq m \
  --device cuda:0 \
  --num_workers 0
```

### 2. 查看预测结果

```bash
# 基础预测分析
python view_predictions.py

# 高级预测分析（包含概率分布和置信区间）
python advanced_prediction_viewer.py

# 快速可视化（推荐）
python quick_visualize.py

# 完整可视化分析
python visualize_predictions.py
```

### 3. 处理重叠预测

```bash
# 演示重叠预测问题
python select_final_predictions.py

# 处理实际预测结果
python overlapping_prediction_handler.py
```

## 配置说明

### 模型参数
- `seq_len`: 输入序列长度（默认36天）
- `pred_len`: 预测长度（默认6天）
- `text_len`: 文本序列长度（默认36天）
- `nsample`: 生成样本数量（默认15个）

### 扩散模型参数
- `num_steps`: 扩散步数（默认500）
- `sample_steps`: 采样步数（默认100）
- `beta_start/beta_end`: 噪声调度参数

## 输出结果

### 预测文件
- `generated_*.npy`: 原始预测样本
- `prediction_median_*.npy`: 中位数预测
- `prediction_mean_*.npy`: 均值预测
- `confidence_50_*.npy`: 50%置信区间
- `confidence_90_*.npy`: 90%置信区间

### 分析报告
- `config_results.json`: 评估指标
- `detailed_predictions.csv`: 详细预测结果
- `prediction_distribution.png`: 预测分布图
- `final_predictions_analysis.json`: 重叠预测分析

## 评估指标

### 二分类任务（涨跌预测）
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 上涨概率

### 连续值任务（收益率预测）
- 均方误差（MSE）
- 平均绝对误差（MAE）
- 正收益率概率
- 置信区间覆盖率

## 高级功能

### 1. 概率分布分析
- 预测值分布统计
- 不确定性量化
- 置信区间分析

### 2. 重叠预测处理
- 自动识别重叠预测
- 多种选择策略
- 一致性分析

### 3. 可视化分析
- **时间序列预测图**: 横轴时间，纵轴OT值，包含50%和90%置信区间
- **预测分布图**: 每个预测点的概率分布直方图
- **置信区间分析**: 置信区间宽度和覆盖范围分析
- **多样本对比**: 多个预测样本的对比可视化

## 可视化使用示例

### 快速可视化
```bash
# 交互式选择可视化类型
python quick_visualize.py

# 直接可视化第一个样本
python -c "from quick_visualize import quick_plot_predictions; quick_plot_predictions(sample_idx=0)"
```

### 完整可视化分析
```bash
# 生成所有可视化图表
python visualize_predictions.py
```

### 可视化输出
- **时间序列图**: 显示历史数据、预测值和置信区间
- **分布图**: 每个预测点的概率分布
- **置信区间分析**: 不确定性量化分析

## 注意事项

1. **数据路径**：确保数据文件路径正确
2. **GPU内存**：大模型可能需要较多GPU内存
3. **训练时间**：完整训练可能需要数小时
4. **数据质量**：确保数值和文本数据的时间对齐
5. **可视化依赖**：需要安装matplotlib和seaborn

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。

## 引用

如果您使用了本项目，请引用相关论文：

```bibtex
@article{mcd_tsf_2024,
  title={Multi-modal Conditional Diffusion for Time Series Forecasting},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 联系方式

如有问题，请通过GitHub Issues联系。
