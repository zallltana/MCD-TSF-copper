#!/usr/bin/env python3
"""
快速可视化脚本 - 简单易用的预测结果可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def quick_plot_predictions(folder_path=None, sample_idx=0, save_path=None):
    """
    快速绘制单个样本的预测结果
    
    Args:
        folder_path: 结果文件夹路径
        sample_idx: 要显示的样本索引
        save_path: 保存路径
    """
    # 自动查找结果文件夹
    if not folder_path:
        result_folders = glob.glob("save/forecasting_copper_*")
        if not result_folders:
            print("未找到预测结果文件夹")
            return
        folder_path = max(result_folders, key=os.path.getctime)
    
    print(f"使用结果文件夹: {folder_path}")
    
    # 加载必要的数据
    files = {
        'median': 'prediction_median_15_guide0.8.npy',
        'ci_50_lower': 'confidence_50_lower_15_guide0.8.npy',
        'ci_50_upper': 'confidence_50_upper_15_guide0.8.npy',
        'ci_90_lower': 'confidence_90_lower_15_guide0.8.npy',
        'ci_90_upper': 'confidence_90_upper_15_guide0.8.npy',
        'target': 'target_15_guide0.8.npy'
    }
    
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            data[key] = np.load(filepath)
        else:
            print(f"警告: 文件不存在 {filepath}")
    
    if 'median' not in data:
        print("错误: 无法加载预测数据")
        return
    
    # 获取数据
    seq_len = 36
    pred_len = 6
    
    # 获取指定样本的数据
    median_pred = data['median'][sample_idx, :, 0]
    ci_50_lower = data['ci_50_lower'][sample_idx, :, 0]
    ci_50_upper = data['ci_50_upper'][sample_idx, :, 0]
    ci_90_lower = data['ci_90_lower'][sample_idx, :, 0]
    ci_90_upper = data['ci_90_upper'][sample_idx, :, 0]
    
    # 分离历史和预测数据
    hist_data = median_pred[:seq_len]
    pred_data = median_pred[seq_len:]
    
    hist_ci_50_lower = ci_50_lower[:seq_len]
    hist_ci_50_upper = ci_50_upper[:seq_len]
    hist_ci_90_lower = ci_90_lower[:seq_len]
    hist_ci_90_upper = ci_90_upper[:seq_len]
    
    pred_ci_50_lower = ci_50_lower[seq_len:]
    pred_ci_50_upper = ci_50_upper[seq_len:]
    pred_ci_90_lower = ci_90_lower[seq_len:]
    pred_ci_90_upper = ci_90_upper[seq_len:]
    
    # 创建日期
    start_date = datetime(2010, 10, 11)
    hist_dates = [start_date + timedelta(days=i) for i in range(seq_len)]
    pred_dates = [start_date + timedelta(days=i) for i in range(seq_len, seq_len + pred_len)]
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制90%置信区间（浅蓝色，透明度0.2）
    plt.fill_between(pred_dates, 
                     pred_ci_90_lower, 
                     pred_ci_90_upper, 
                     alpha=0.2, 
                     color='lightblue', 
                     label='90% 置信区间')
    
    # 绘制50%置信区间（深蓝色，透明度0.4）
    plt.fill_between(pred_dates, 
                     pred_ci_50_lower, 
                     pred_ci_50_upper, 
                     alpha=0.4, 
                     color='blue', 
                     label='50% 置信区间')
    
    # 绘制历史数据
    plt.plot(hist_dates, hist_data, 
             color='black', linewidth=2, label='历史数据')
    
    # 绘制预测数据
    plt.plot(pred_dates, pred_data, 
             color='red', linewidth=2, linestyle='--', marker='o', 
             markersize=6, label='预测值')
    
    # 显示实际值（如果有）
    if 'target' in data:
        actual_data = data['target'][sample_idx, :, 0]
        actual_hist = actual_data[:seq_len]
        actual_pred = actual_data[seq_len:]
        
        plt.plot(hist_dates, actual_hist, 
                 color='gray', linewidth=1, alpha=0.7, label='历史实际值')
        plt.plot(pred_dates, actual_pred, 
                 color='green', linewidth=2, marker='s', 
                 markersize=6, label='实际值')
    
    # 添加分隔线
    plt.axvline(x=hist_dates[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    # 设置标题和标签
    plt.title(f'样本 {sample_idx + 1} - 铜价预测结果 (MCD-TSF)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('OT值', fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()

def plot_multiple_samples(folder_path=None, sample_indices=[0, 1, 2], save_path=None):
    """绘制多个样本的预测结果"""
    if not folder_path:
        result_folders = glob.glob("save/forecasting_copper_*")
        if not result_folders:
            print("未找到预测结果文件夹")
            return
        folder_path = max(result_folders, key=os.path.getctime)
    
    # 创建子图
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(15, 5*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx]
        
        # 加载数据（简化版）
        median_file = os.path.join(folder_path, 'prediction_median_15_guide0.8.npy')
        ci_50_lower_file = os.path.join(folder_path, 'confidence_50_lower_15_guide0.8.npy')
        ci_50_upper_file = os.path.join(folder_path, 'confidence_50_upper_15_guide0.8.npy')
        ci_90_lower_file = os.path.join(folder_path, 'confidence_90_lower_15_guide0.8.npy')
        ci_90_upper_file = os.path.join(folder_path, 'confidence_90_upper_15_guide0.8.npy')
        
        if not all(os.path.exists(f) for f in [median_file, ci_50_lower_file, ci_50_upper_file, ci_90_lower_file, ci_90_upper_file]):
            print(f"警告: 样本 {sample_idx} 的数据文件不完整")
            continue
        
        # 加载数据
        median_pred = np.load(median_file)[sample_idx, :, 0]
        ci_50_lower = np.load(ci_50_lower_file)[sample_idx, :, 0]
        ci_50_upper = np.load(ci_50_upper_file)[sample_idx, :, 0]
        ci_90_lower = np.load(ci_90_lower_file)[sample_idx, :, 0]
        ci_90_upper = np.load(ci_90_upper_file)[sample_idx, :, 0]
        
        # 分离数据
        seq_len = 36
        pred_len = 6
        
        hist_data = median_pred[:seq_len]
        pred_data = median_pred[seq_len:]
        
        pred_ci_50_lower = ci_50_lower[seq_len:]
        pred_ci_50_upper = ci_50_upper[seq_len:]
        pred_ci_90_lower = ci_90_lower[seq_len:]
        pred_ci_90_upper = ci_90_upper[seq_len:]
        
        # 创建日期
        start_date = datetime(2010, 10, 11)
        hist_dates = [start_date + timedelta(days=i) for i in range(seq_len)]
        pred_dates = [start_date + timedelta(days=i) for i in range(seq_len, seq_len + pred_len)]
        
        # 绘制置信区间
        ax.fill_between(pred_dates, pred_ci_90_lower, pred_ci_90_upper, 
                       alpha=0.2, color='lightblue', label='90% 置信区间')
        ax.fill_between(pred_dates, pred_ci_50_lower, pred_ci_50_upper, 
                       alpha=0.4, color='blue', label='50% 置信区间')
        
        # 绘制数据线
        ax.plot(hist_dates, hist_data, color='black', linewidth=2, label='历史数据')
        ax.plot(pred_dates, pred_data, color='red', linewidth=2, linestyle='--', 
               marker='o', markersize=4, label='预测值')
        
        # 添加分隔线
        ax.axvline(x=hist_dates[-1], color='gray', linestyle=':', alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(f'样本 {sample_idx + 1} - 铜价预测结果', fontweight='bold')
        ax.set_ylabel('OT值')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 设置x轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('MCD-TSF 铜价预测结果 - 多样本对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多样本图像已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("=== MCD-TSF 预测结果快速可视化 ===")
    print("1. 绘制单个样本预测结果")
    print("2. 绘制多个样本对比")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        sample_idx = int(input("请输入样本索引 (0开始): ") or "0")
        quick_plot_predictions(sample_idx=sample_idx)
    elif choice == "2":
        samples = input("请输入样本索引列表 (用逗号分隔，如 0,1,2): ").strip()
        if samples:
            sample_indices = [int(x.strip()) for x in samples.split(',')]
        else:
            sample_indices = [0, 1, 2]
        plot_multiple_samples(sample_indices=sample_indices)
    else:
        print("无效选择，使用默认设置...")
        quick_plot_predictions(sample_idx=0)
