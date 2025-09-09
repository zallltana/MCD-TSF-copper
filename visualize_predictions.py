#!/usr/bin/env python3
"""
时间序列预测结果可视化工具
绘制预测结果、置信区间和概率分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import os
import glob
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PredictionVisualizer:
    def __init__(self, folder_path=None):
        self.folder_path = folder_path
        self.results = None
        self.date_range = None
        
    def load_prediction_results(self, folder_path=None):
        """加载预测结果"""
        if folder_path:
            self.folder_path = folder_path
        
        if not self.folder_path:
            # 自动查找最新的结果文件夹
            result_folders = glob.glob("save/forecasting_copper_*")
            if not result_folders:
                print("未找到预测结果文件夹")
                return False
            self.folder_path = max(result_folders, key=os.path.getctime)
        
        print(f"加载结果文件夹: {self.folder_path}")
        
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
        }
        
        self.results = {}
        for key, filename in files.items():
            filepath = os.path.join(self.folder_path, filename)
            if os.path.exists(filepath):
                self.results[key] = np.load(filepath)
                print(f"加载 {key}: {self.results[key].shape}")
            else:
                print(f"文件不存在: {filepath}")
        
        return len(self.results) > 0
    
    def create_date_range(self, start_date='2010-10-11', end_date='2022-09-15'):
        """创建日期范围"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 创建完整的日期序列
        date_list = []
        current = start
        while current <= end:
            date_list.append(current)
            current += timedelta(days=1)
        
        self.date_range = date_list
        return date_list
    
    def plot_time_series_with_confidence(self, 
                                       sample_indices=None, 
                                       save_path=None,
                                       figsize=(15, 8),
                                       show_actual=True,
                                       show_individual_predictions=False):
        """
        绘制时间序列预测结果，包含置信区间
        
        Args:
            sample_indices: 要显示的样本索引列表，None表示显示所有
            save_path: 保存路径
            figsize: 图像大小
            show_actual: 是否显示实际值
            show_individual_predictions: 是否显示个别预测样本
        """
        if not self.results:
            print("请先加载预测结果")
            return
        
        # 创建日期范围
        if not self.date_range:
            self.create_date_range()
        
        # 获取数据维度
        batch_size, seq_len_pred_len, _ = self.results['median'].shape
        seq_len = 36
        pred_len = 6
        
        # 选择要显示的样本
        if sample_indices is None:
            sample_indices = list(range(min(10, batch_size)))  # 默认显示前10个样本
        
        # 创建图形
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=figsize)
        if len(sample_indices) == 1:
            axes = [axes]
        
        for idx, sample_idx in enumerate(sample_indices):
            ax = axes[idx]
            
            # 获取该样本的数据
            sample_median = self.results['median'][sample_idx, :, 0]
            sample_ci_50_lower = self.results['ci_50_lower'][sample_idx, :, 0]
            sample_ci_50_upper = self.results['ci_50_upper'][sample_idx, :, 0]
            sample_ci_90_lower = self.results['ci_90_lower'][sample_idx, :, 0]
            sample_ci_90_upper = self.results['ci_90_upper'][sample_idx, :, 0]
            
            # 分离历史数据和预测数据
            historical_data = sample_median[:seq_len]
            prediction_data = sample_median[seq_len:]
            
            historical_ci_50_lower = sample_ci_50_lower[:seq_len]
            historical_ci_50_upper = sample_ci_50_upper[:seq_len]
            historical_ci_90_lower = sample_ci_90_lower[:seq_len]
            historical_ci_90_upper = sample_ci_90_upper[:seq_len]
            
            pred_ci_50_lower = sample_ci_50_lower[seq_len:]
            pred_ci_50_upper = sample_ci_50_upper[seq_len:]
            pred_ci_90_lower = sample_ci_90_lower[seq_len:]
            pred_ci_90_upper = sample_ci_90_upper[seq_len:]
            
            # 创建对应的日期
            hist_dates = self.date_range[:seq_len]
            pred_dates = self.date_range[seq_len:seq_len+pred_len]
            
            # 绘制90%置信区间（浅色）
            ax.fill_between(pred_dates, 
                           pred_ci_90_lower, 
                           pred_ci_90_upper, 
                           alpha=0.2, 
                           color='lightblue', 
                           label='90% 置信区间')
            
            # 绘制50%置信区间（深色）
            ax.fill_between(pred_dates, 
                           pred_ci_50_lower, 
                           pred_ci_50_upper, 
                           alpha=0.4, 
                           color='blue', 
                           label='50% 置信区间')
            
            # 绘制历史数据
            ax.plot(hist_dates, historical_data, 
                   color='black', linewidth=2, label='历史数据')
            
            # 绘制预测数据
            ax.plot(pred_dates, prediction_data, 
                   color='red', linewidth=2, linestyle='--', label='预测值')
            
            # 显示实际值（如果有）
            if show_actual and 'target' in self.results:
                actual_data = self.results['target'][sample_idx, :, 0]
                actual_hist = actual_data[:seq_len]
                actual_pred = actual_data[seq_len:]
                
                ax.plot(hist_dates, actual_hist, 
                       color='gray', linewidth=1, alpha=0.7, label='历史实际值')
                ax.plot(pred_dates, actual_pred, 
                       color='green', linewidth=2, marker='o', markersize=4, label='实际值')
            
            # 显示个别预测样本（可选）
            if show_individual_predictions and 'generated' in self.results:
                generated_samples = self.results['generated'][sample_idx, :, seq_len:, 0]
                for i in range(min(5, generated_samples.shape[0])):  # 只显示前5个样本
                    ax.plot(pred_dates, generated_samples[i, :], 
                           color='lightgray', alpha=0.3, linewidth=0.5)
            
            # 添加垂直线分隔历史和预测
            ax.axvline(x=hist_dates[-1], color='gray', linestyle=':', alpha=0.7)
            
            # 设置标题和标签
            ax.set_title(f'样本 {sample_idx + 1} - 铜价预测结果', fontsize=12, fontweight='bold')
            ax.set_ylabel('OT值', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 设置x轴格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 设置总标题
        fig.suptitle('MCD-TSF 铜价预测结果 - 时间序列可视化', fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, save_path=None, figsize=(12, 8)):
        """绘制预测分布图"""
        if not self.results or 'generated' not in self.results:
            print("请先加载预测结果")
            return
        
        generated = self.results['generated']
        seq_len = 36
        pred_len = 6
        
        # 只分析预测期
        future_samples = generated[:, :, seq_len:, 0]  # (batch_size, n_samples, pred_len)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('预测分布分析 - 未来6天', fontsize=16, fontweight='bold')
        
        # 绘制每个预测点的分布
        for i in range(min(6, future_samples.shape[2])):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 获取第i个预测点的所有样本
            day_samples = future_samples[:, :, i].flatten()
            
            # 绘制直方图
            ax.hist(day_samples, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # 添加统计线
            ax.axvline(np.median(day_samples), color='red', linestyle='--', linewidth=2, label='中位数')
            ax.axvline(np.mean(day_samples), color='green', linestyle='--', linewidth=2, label='均值')
            
            # 添加置信区间
            ci_50_lower = np.percentile(day_samples, 25)
            ci_50_upper = np.percentile(day_samples, 75)
            ci_90_lower = np.percentile(day_samples, 5)
            ci_90_upper = np.percentile(day_samples, 95)
            
            ax.axvspan(ci_50_lower, ci_50_upper, alpha=0.3, color='blue', label='50%置信区间')
            ax.axvspan(ci_90_lower, ci_90_upper, alpha=0.2, color='gray', label='90%置信区间')
            
            # 设置标题和标签
            ax.set_title(f'第{i+1}天预测分布', fontweight='bold')
            ax.set_xlabel('预测值')
            ax.set_ylabel('密度')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_confidence_interval_analysis(self, save_path=None, figsize=(12, 6)):
        """绘制置信区间分析图"""
        if not self.results:
            print("请先加载预测结果")
            return
        
        # 计算置信区间宽度
        ci_50_width = self.results['ci_50_upper'] - self.results['ci_50_lower']
        ci_90_width = self.results['ci_90_upper'] - self.results['ci_90_lower']
        
        # 只分析预测期
        seq_len = 36
        pred_len = 6
        
        pred_ci_50_width = ci_50_width[:, seq_len:, 0]
        pred_ci_90_width = ci_90_width[:, seq_len:, 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 绘制置信区间宽度分布
        ax1.hist(pred_ci_50_width.flatten(), bins=30, alpha=0.7, label='50%置信区间', color='blue')
        ax1.hist(pred_ci_90_width.flatten(), bins=30, alpha=0.7, label='90%置信区间', color='red')
        ax1.set_xlabel('置信区间宽度')
        ax1.set_ylabel('频次')
        ax1.set_title('置信区间宽度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制不同预测天的置信区间宽度
        days = range(1, pred_len + 1)
        mean_ci_50 = [np.mean(pred_ci_50_width[:, i]) for i in range(pred_len)]
        mean_ci_90 = [np.mean(pred_ci_90_width[:, i]) for i in range(pred_len)]
        
        ax2.plot(days, mean_ci_50, 'o-', label='50%置信区间', color='blue', linewidth=2)
        ax2.plot(days, mean_ci_90, 's-', label='90%置信区间', color='red', linewidth=2)
        ax2.set_xlabel('预测天数')
        ax2.set_ylabel('平均置信区间宽度')
        ax2.set_title('不同预测天的置信区间宽度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(days)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"置信区间分析图已保存到: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    # 创建可视化器
    visualizer = PredictionVisualizer()
    
    # 加载预测结果
    if not visualizer.load_prediction_results():
        print("无法加载预测结果")
        return
    
    # 创建日期范围
    visualizer.create_date_range()
    
    # 绘制时间序列预测结果
    print("绘制时间序列预测结果...")
    visualizer.plot_time_series_with_confidence(
        sample_indices=[0, 1, 2],  # 显示前3个样本
        save_path=os.path.join(visualizer.folder_path, "time_series_predictions.png"),
        show_actual=True,
        show_individual_predictions=False
    )
    
    # 绘制预测分布图
    print("绘制预测分布图...")
    visualizer.plot_prediction_distribution(
        save_path=os.path.join(visualizer.folder_path, "prediction_distributions.png")
    )
    
    # 绘制置信区间分析图
    print("绘制置信区间分析图...")
    visualizer.plot_confidence_interval_analysis(
        save_path=os.path.join(visualizer.folder_path, "confidence_interval_analysis.png")
    )
    
    print("所有可视化完成！")

if __name__ == "__main__":
    main()
