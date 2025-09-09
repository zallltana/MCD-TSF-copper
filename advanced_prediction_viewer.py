#!/usr/bin/env python3
"""
高级预测结果查看器
支持不同OT类型（二分类、连续值）和概率分布分析
"""

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def load_prediction_results(folder_path):
    """加载预测结果"""
    # 查找最新的结果文件夹
    result_folders = glob.glob(os.path.join(folder_path, "save/forecasting_copper_*"))
    if not result_folders:
        print("未找到预测结果文件夹")
        return None
    
    latest_folder = max(result_folders, key=os.path.getctime)
    print(f"使用结果文件夹: {latest_folder}")
    
    # 加载预测结果
    files = {
        'generated': 'generated_nsample15_guide0.8.npy',
        'target': 'target_15_guide0.8.npy', 
        'median': 'prediction_median_15_guide0.8.npy',
        'mean': 'prediction_mean_15_guide0.8.npy',
        'std': 'prediction_std_15_guide0.8.npy',
        'ci_50_lower': 'confidence_50_lower_15_guide0.8.npy',
        'ci_50_upper': 'confidence_50_upper_15_guide0.8.npy',
        'ci_90_lower': 'confidence_90_lower_15_guide0.8.npy',
        'ci_90_upper': 'confidence_90_upper_15_guide0.8.npy',
        'up_prob': 'up_probability_15_guide0.8.npy',
        'positive_prob': 'positive_return_prob_15_guide0.8.npy'
    }
    
    results = {}
    for key, filename in files.items():
        filepath = os.path.join(latest_folder, filename)
        if os.path.exists(filepath):
            results[key] = np.load(filepath)
            print(f"加载 {key}: {results[key].shape}")
        else:
            print(f"文件不存在: {filepath}")
    
    return results, latest_folder

def analyze_probability_distribution(results):
    """分析概率分布"""
    if not results or 'generated' not in results:
        return
    
    generated = results['generated']  # (batch_size, n_samples, seq_len+pred_len, 1)
    seq_len = 36
    pred_len = 6
    
    # 只分析预测期
    future_samples = generated[:, :, seq_len:, 0]  # (batch_size, n_samples, pred_len)
    
    print(f"\n=== 概率分布分析 ===")
    print(f"样本形状: {future_samples.shape}")
    print(f"每个预测点基于 {future_samples.shape[1]} 个样本")
    
    # 计算整体统计
    all_samples = future_samples.flatten()
    print(f"\n整体统计:")
    print(f"  均值: {np.mean(all_samples):.4f}")
    print(f"  标准差: {np.std(all_samples):.4f}")
    print(f"  最小值: {np.min(all_samples):.4f}")
    print(f"  最大值: {np.max(all_samples):.4f}")
    print(f"  25%分位数: {np.percentile(all_samples, 25):.4f}")
    print(f"  50%分位数: {np.percentile(all_samples, 50):.4f}")
    print(f"  75%分位数: {np.percentile(all_samples, 75):.4f}")
    print(f"  95%分位数: {np.percentile(all_samples, 95):.4f}")
    print(f"  5%分位数: {np.percentile(all_samples, 5):.4f}")

def analyze_confidence_intervals(results):
    """分析置信区间"""
    if not results:
        return
    
    print(f"\n=== 置信区间分析 ===")
    
    if 'ci_50_lower' in results and 'ci_50_upper' in results:
        ci_50_lower = results['ci_50_lower']
        ci_50_upper = results['ci_50_upper']
        
        print(f"50%置信区间:")
        print(f"  平均下界: {np.mean(ci_50_lower):.4f}")
        print(f"  平均上界: {np.mean(ci_50_upper):.4f}")
        print(f"  平均区间宽度: {np.mean(ci_50_upper - ci_50_lower):.4f}")
    
    if 'ci_90_lower' in results and 'ci_90_upper' in results:
        ci_90_lower = results['ci_90_lower']
        ci_90_upper = results['ci_90_upper']
        
        print(f"90%置信区间:")
        print(f"  平均下界: {np.mean(ci_90_lower):.4f}")
        print(f"  平均上界: {np.mean(ci_90_upper):.4f}")
        print(f"  平均区间宽度: {np.mean(ci_90_upper - ci_90_lower):.4f}")

def analyze_probability_metrics(results):
    """分析概率指标"""
    if not results:
        return
    
    print(f"\n=== 概率指标分析 ===")
    
    if 'up_prob' in results:
        up_prob = results['up_prob']
        print(f"上涨概率 (二分类任务):")
        print(f"  平均上涨概率: {np.mean(up_prob):.4f}")
        print(f"  上涨概率标准差: {np.std(up_prob):.4f}")
        print(f"  高置信度上涨 (>0.7): {np.sum(up_prob > 0.7)} 个预测点")
        print(f"  高置信度下跌 (<0.3): {np.sum(up_prob < 0.3)} 个预测点")
    
    if 'positive_prob' in results:
        positive_prob = results['positive_prob']
        print(f"正收益率概率 (连续值任务):")
        print(f"  平均正收益率概率: {np.mean(positive_prob):.4f}")
        print(f"  正收益率概率标准差: {np.std(positive_prob):.4f}")
        print(f"  高置信度正收益 (>0.7): {np.sum(positive_prob > 0.7)} 个预测点")
        print(f"  高置信度负收益 (<0.3): {np.sum(positive_prob < 0.3)} 个预测点")

def show_detailed_predictions(results, n_samples=5):
    """显示详细预测结果"""
    if not results or 'median' not in results:
        return
    
    median_pred = results['median']  # (batch_size, pred_len, 1)
    target = results.get('target', None)
    
    if target is not None:
        target = target[:, 36:, 0]  # 只取预测期
    
    print(f"\n=== 详细预测结果 (前{n_samples}个样本) ===")
    print("样本 | 预测期预测值 | 预测期实际值 | 50%置信区间 | 90%置信区间")
    print("-" * 80)
    
    for i in range(min(n_samples, median_pred.shape[0])):
        pred_values = median_pred[i, :, 0]
        pred_str = " ".join([f"{x:.3f}" for x in pred_values])
        
        if target is not None:
            target_values = target[i, :]
            target_str = " ".join([f"{x:.3f}" for x in target_values])
        else:
            target_str = "N/A"
        
        # 置信区间
        if 'ci_50_lower' in results and 'ci_50_upper' in results:
            ci_50_lower = results['ci_50_lower'][i, :, 0]
            ci_50_upper = results['ci_50_upper'][i, :, 0]
            ci_50_str = f"[{ci_50_lower[0]:.3f}, {ci_50_upper[0]:.3f}]"
        else:
            ci_50_str = "N/A"
        
        if 'ci_90_lower' in results and 'ci_90_upper' in results:
            ci_90_lower = results['ci_90_lower'][i, :, 0]
            ci_90_upper = results['ci_90_upper'][i, :, 0]
            ci_90_str = f"[{ci_90_lower[0]:.3f}, {ci_90_upper[0]:.3f}]"
        else:
            ci_90_str = "N/A"
        
        print(f"{i+1:4d} | {pred_str:12s} | {target_str:12s} | {ci_50_str:12s} | {ci_90_str:12s}")

def plot_prediction_distribution(results, save_path=None):
    """绘制预测分布图"""
    if not results or 'generated' not in results:
        return
    
    generated = results['generated']
    seq_len = 36
    future_samples = generated[:, :, seq_len:, 0]  # (batch_size, n_samples, pred_len)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('预测分布分析', fontsize=16)
    
    # 绘制每个预测点的分布
    for i in range(min(6, future_samples.shape[2])):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 获取第i个预测点的所有样本
        day_samples = future_samples[:, :, i].flatten()
        
        # 绘制直方图
        ax.hist(day_samples, bins=30, alpha=0.7, density=True)
        ax.axvline(np.median(day_samples), color='red', linestyle='--', label='中位数')
        ax.axvline(np.mean(day_samples), color='green', linestyle='--', label='均值')
        
        # 添加置信区间
        ci_50_lower = np.percentile(day_samples, 25)
        ci_50_upper = np.percentile(day_samples, 75)
        ci_90_lower = np.percentile(day_samples, 5)
        ci_90_upper = np.percentile(day_samples, 95)
        
        ax.axvspan(ci_50_lower, ci_50_upper, alpha=0.3, color='blue', label='50%置信区间')
        ax.axvspan(ci_90_lower, ci_90_upper, alpha=0.2, color='gray', label='90%置信区间')
        
        ax.set_title(f'第{i+1}天预测分布')
        ax.set_xlabel('预测值')
        ax.set_ylabel('密度')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到: {save_path}")
    
    plt.show()

def export_predictions_to_csv(results, save_path):
    """导出预测结果到CSV"""
    if not results or 'median' not in results:
        return
    
    median_pred = results['median']
    batch_size, pred_len, _ = median_pred.shape
    
    # 创建DataFrame
    data = []
    for i in range(batch_size):
        for j in range(pred_len):
            row = {
                'sample_id': i,
                'day': j + 1,
                'prediction_median': median_pred[i, j, 0],
                'prediction_mean': results.get('mean', np.zeros_like(median_pred))[i, j, 0],
                'prediction_std': results.get('std', np.zeros_like(median_pred))[i, j, 0],
            }
            
            # 添加置信区间
            if 'ci_50_lower' in results:
                row['ci_50_lower'] = results['ci_50_lower'][i, j, 0]
                row['ci_50_upper'] = results['ci_50_upper'][i, j, 0]
            
            if 'ci_90_lower' in results:
                row['ci_90_lower'] = results['ci_90_lower'][i, j, 0]
                row['ci_90_upper'] = results['ci_90_upper'][i, j, 0]
            
            # 添加概率
            if 'up_prob' in results:
                row['up_probability'] = results['up_prob'][i, j, 0]
            
            if 'positive_prob' in results:
                row['positive_return_probability'] = results['positive_prob'][i, j, 0]
            
            # 添加实际值
            if 'target' in results:
                row['actual_value'] = results['target'][i, 36+j, 0]
            
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"预测结果已导出到: {save_path}")

if __name__ == "__main__":
    # 加载结果
    results, folder_path = load_prediction_results(".")
    
    if results:
        # 分析概率分布
        analyze_probability_distribution(results)
        
        # 分析置信区间
        analyze_confidence_intervals(results)
        
        # 分析概率指标
        analyze_probability_metrics(results)
        
        # 显示详细预测
        show_detailed_predictions(results)
        
        # 绘制分布图
        plot_prediction_distribution(results, os.path.join(folder_path, "prediction_distribution.png"))
        
        # 导出到CSV
        export_predictions_to_csv(results, os.path.join(folder_path, "detailed_predictions.csv"))
        
        print(f"\n所有结果保存在: {folder_path}")
        print("生成的文件:")
        print("- prediction_distribution.png: 预测分布图")
        print("- detailed_predictions.csv: 详细预测结果")
        print("- 各种.npy文件: 原始预测数据")
