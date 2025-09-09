#!/usr/bin/env python3
"""
简化版最终预测选择器
演示如何处理重叠预测问题
"""

import numpy as np
import pandas as pd
import os
import glob

def demonstrate_overlapping_problem():
    """演示重叠预测问题"""
    print("=== 重叠预测问题演示 ===")
    print("假设我们有以下预测序列：")
    print()
    
    # 模拟数据
    dates = ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6']
    
    predictions = {
        '预测1 (1.1-1.30数据)': [0.6, 0.7, 0.5, 0.8, 0.4, 0.9],
        '预测2 (1.2-1.31数据)': [0.7, 0.6, 0.8, 0.5, 0.9, 0.3],
        '预测3 (1.3-2.1数据)': [0.5, 0.8, 0.6, 0.7, 0.3, 0.8],
        '预测4 (1.4-2.2数据)': [0.8, 0.5, 0.7, 0.6, 0.8, 0.4],
        '预测5 (1.5-2.3数据)': [0.4, 0.9, 0.3, 0.8, 0.6, 0.7],
        '预测6 (1.6-2.4数据)': [0.9, 0.3, 0.8, 0.4, 0.7, 0.5]
    }
    
    # 创建DataFrame显示
    df = pd.DataFrame(predictions, index=dates)
    print(df.round(3))
    print()
    
    print("问题：2.2这个日期有6个不同的预测值！")
    print("2.2的预测值:", [pred[1] for pred in predictions.values()])
    print()

def select_final_predictions_simple(predictions_dict, method='latest'):
    """简单的最终预测选择方法"""
    dates = list(predictions_dict.keys())
    pred_names = list(predictions_dict[dates[0]].keys())
    
    final_predictions = {}
    
    for i, date in enumerate(dates):
        # 收集该日期的所有预测
        date_predictions = []
        for pred_name in pred_names:
            date_predictions.append(predictions_dict[date][pred_name])
        
        if method == 'latest':
            # 选择最新的预测（最后一个）
            final_pred = date_predictions[-1]
            selected_method = f"最新预测 (预测{len(pred_names)})"
        
        elif method == 'average':
            # 选择平均值
            final_pred = np.mean(date_predictions)
            selected_method = "平均值"
        
        elif method == 'weighted':
            # 加权平均，越新的权重越高
            weights = np.arange(1, len(date_predictions) + 1)
            weights = weights / weights.sum()
            final_pred = np.average(date_predictions, weights=weights)
            selected_method = "加权平均"
        
        elif method == 'median':
            # 选择中位数
            final_pred = np.median(date_predictions)
            selected_method = "中位数"
        
        final_predictions[date] = {
            'prediction': final_pred,
            'method': selected_method,
            'all_predictions': date_predictions,
            'std': np.std(date_predictions)
        }
    
    return final_predictions

def analyze_prediction_quality(predictions_dict):
    """分析预测质量"""
    print("=== 预测质量分析 ===")
    
    dates = list(predictions_dict.keys())
    pred_names = list(predictions_dict[dates[0]].keys())
    
    for date in dates:
        date_predictions = [predictions_dict[date][pred] for pred in pred_names]
        
        print(f"\n{date}的预测分析:")
        print(f"  所有预测值: {[f'{p:.3f}' for p in date_predictions]}")
        print(f"  均值: {np.mean(date_predictions):.3f}")
        print(f"  标准差: {np.std(date_predictions):.3f}")
        print(f"  范围: {np.max(date_predictions) - np.min(date_predictions):.3f}")
        print(f"  变异系数: {np.std(date_predictions)/np.mean(date_predictions):.3f}")
        
        # 判断预测一致性
        if np.std(date_predictions) < 0.1:
            consistency = "高一致性"
        elif np.std(date_predictions) < 0.2:
            consistency = "中等一致性"
        else:
            consistency = "低一致性"
        print(f"  一致性: {consistency}")

def main():
    """主函数"""
    # 演示重叠预测问题
    demonstrate_overlapping_problem()
    
    # 创建示例数据
    predictions_dict = {
        '2.1': {'预测1': 0.6, '预测2': 0.7, '预测3': 0.5, '预测4': 0.8, '预测5': 0.4, '预测6': 0.9},
        '2.2': {'预测1': 0.7, '预测2': 0.6, '预测3': 0.8, '预测4': 0.5, '预测5': 0.9, '预测6': 0.3},
        '2.3': {'预测1': 0.5, '预测2': 0.8, '预测3': 0.6, '预测4': 0.7, '预测5': 0.3, '预测6': 0.8},
        '2.4': {'预测1': 0.8, '预测2': 0.5, '预测3': 0.7, '预测4': 0.6, '预测5': 0.8, '预测6': 0.4},
        '2.5': {'预测1': 0.4, '预测2': 0.9, '预测3': 0.3, '预测4': 0.8, '预测5': 0.6, '预测6': 0.7},
        '2.6': {'预测1': 0.9, '预测2': 0.3, '预测3': 0.8, '预测4': 0.4, '预测5': 0.7, '预测6': 0.5}
    }
    
    # 分析预测质量
    analyze_prediction_quality(predictions_dict)
    
    # 测试不同方法
    methods = ['latest', 'average', 'weighted', 'median']
    
    print("\n=== 不同方法的结果比较 ===")
    print("日期  | 最新预测 | 平均值  | 加权平均 | 中位数")
    print("-" * 50)
    
    for date in predictions_dict.keys():
        latest = select_final_predictions_simple(predictions_dict, 'latest')[date]['prediction']
        average = select_final_predictions_simple(predictions_dict, 'average')[date]['prediction']
        weighted = select_final_predictions_simple(predictions_dict, 'weighted')[date]['prediction']
        median = select_final_predictions_simple(predictions_dict, 'median')[date]['prediction']
        
        print(f"{date:4s} | {latest:8.3f} | {average:8.3f} | {weighted:8.3f} | {median:8.3f}")
    
    print("\n=== 推荐方法 ===")
    print("1. 最新预测 (Latest): 使用最新的预测，适合快速变化的市场")
    print("2. 平均值 (Average): 使用所有预测的平均值，减少噪声")
    print("3. 加权平均 (Weighted): 越新的预测权重越高，平衡稳定性和时效性")
    print("4. 中位数 (Median): 对异常值不敏感，更稳健")
    print("\n建议：对于金融预测，推荐使用加权平均或最新预测方法")

if __name__ == "__main__":
    main()
