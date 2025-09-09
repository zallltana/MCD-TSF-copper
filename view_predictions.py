#!/usr/bin/env python3
"""
查看铜价预测结果的脚本
用于分析0/1分类预测结果
"""

import numpy as np
import pandas as pd
import os
import glob

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
        'binary_predictions': 'binary_predictions_15_guide0.8.npy',
        'binary_targets': 'binary_targets_15_guide0.8.npy'
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

def analyze_predictions(results):
    """分析预测结果"""
    if not results:
        return
    
    # 获取预测期数据（后6个时间步）
    pred_len = 6
    seq_len = 36
    
    # 连续预测值（所有样本的均值）
    if 'generated' in results:
        generated = results['generated']  # (batch_size, n_samples, seq_len+pred_len, 1)
        continuous_pred = generated.mean(axis=1)  # 取所有样本的均值
        future_pred = continuous_pred[:, seq_len:, 0]  # 只取预测期
        print(f"\n连续预测值形状: {future_pred.shape}")
        print(f"预测期连续值范围: [{future_pred.min():.4f}, {future_pred.max():.4f}]")
    
    # 二分类预测
    if 'binary_predictions' in results and 'binary_targets' in results:
        binary_pred = results['binary_predictions']  # (batch_size, seq_len+pred_len, 1)
        binary_target = results['binary_targets']
        
        # 只分析预测期
        future_binary_pred = binary_pred[:, seq_len:, 0]
        future_binary_target = binary_target[:, seq_len:, 0]
        
        print(f"\n二分类预测形状: {future_binary_pred.shape}")
        
        # 计算准确率
        correct = (future_binary_pred == future_binary_target).sum()
        total = future_binary_pred.size
        accuracy = correct / total
        print(f"预测期准确率: {accuracy:.4f} ({correct}/{total})")
        
        # 分析预测分布
        pred_0_count = (future_binary_pred == 0).sum()
        pred_1_count = (future_binary_pred == 1).sum()
        target_0_count = (future_binary_target == 0).sum()
        target_1_count = (future_binary_target == 1).sum()
        
        print(f"\n预测分布:")
        print(f"  预测为0: {pred_0_count} ({pred_0_count/total:.2%})")
        print(f"  预测为1: {pred_1_count} ({pred_1_count/total:.2%})")
        print(f"实际分布:")
        print(f"  实际为0: {target_0_count} ({target_0_count/total:.2%})")
        print(f"  实际为1: {target_1_count} ({target_1_count/total:.2%})")
        
        # 混淆矩阵
        tp = ((future_binary_pred == 1) & (future_binary_target == 1)).sum()
        tn = ((future_binary_pred == 0) & (future_binary_target == 0)).sum()
        fp = ((future_binary_pred == 1) & (future_binary_target == 0)).sum()
        fn = ((future_binary_pred == 0) & (future_binary_target == 1)).sum()
        
        print(f"\n混淆矩阵:")
        print(f"  真正例(TP): {tp}")
        print(f"  真负例(TN): {tn}")
        print(f"  假正例(FP): {fp}")
        print(f"  假负例(FN): {fn}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"  精确率: {precision:.4f}")
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"  召回率: {recall:.4f}")
        
        if tp + fp > 0 and tp + fn > 0:
            f1 = 2 * precision * recall / (precision + recall)
            print(f"  F1分数: {f1:.4f}")

def show_sample_predictions(results, n_samples=5):
    """显示样本预测结果"""
    if 'binary_predictions' not in results or 'binary_targets' not in results:
        return
    
    binary_pred = results['binary_predictions']
    binary_target = results['binary_targets']
    
    seq_len = 36
    pred_len = 6
    
    print(f"\n前{n_samples}个样本的预测结果:")
    print("样本 | 预测期预测 | 预测期实际 | 准确率")
    print("-" * 50)
    
    for i in range(min(n_samples, binary_pred.shape[0])):
        future_pred = binary_pred[i, seq_len:, 0]
        future_target = binary_target[i, seq_len:, 0]
        sample_acc = (future_pred == future_target).mean()
        
        pred_str = "".join([str(int(x)) for x in future_pred])
        target_str = "".join([str(int(x)) for x in future_target])
        
        print(f"{i+1:4d} | {pred_str:10s} | {target_str:10s} | {sample_acc:.3f}")

if __name__ == "__main__":
    # 加载结果
    results, folder_path = load_prediction_results(".")
    
    if results:
        # 分析预测结果
        analyze_predictions(results)
        
        # 显示样本预测
        show_sample_predictions(results)
        
        print(f"\n详细结果保存在: {folder_path}")
        print("可以查看以下文件:")
        print("- binary_predictions_*.npy: 二分类预测结果")
        print("- binary_targets_*.npy: 真实标签")
        print("- generated_*.npy: 连续预测值")
        print("- config_results.json: 评估指标")
