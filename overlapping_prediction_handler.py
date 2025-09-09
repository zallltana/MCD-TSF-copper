#!/usr/bin/env python3
"""
重叠预测处理器
处理滑动窗口预测中的重叠预测问题，选择最终预测结果
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class OverlappingPredictionHandler:
    def __init__(self, seq_len=36, pred_len=6):
        self.seq_len = seq_len
        self.pred_len = pred_len
    
    def extract_date_mappings(self, data_length):
        """提取每个预测样本对应的实际日期映射"""
        mappings = []
        
        for i in range(data_length - self.seq_len - self.pred_len + 1):
            # 输入序列：i 到 i+seq_len-1
            # 预测序列：i+seq_len 到 i+seq_len+pred_len-1
            input_start = i
            input_end = i + self.seq_len - 1
            pred_start = i + self.seq_len
            pred_end = i + self.seq_len + self.pred_len - 1
            
            # 预测的6个日期
            pred_dates = list(range(pred_start, pred_end + 1))
            
            mappings.append({
                'sample_id': i,
                'input_range': (input_start, input_end),
                'prediction_range': (pred_start, pred_end),
                'prediction_dates': pred_dates
            })
        
        return mappings
    
    def resolve_overlapping_predictions(self, predictions, targets=None, method='latest'):
        """
        解决重叠预测问题
        
        Args:
            predictions: (n_samples, pred_len, 1) 预测结果
            targets: (n_samples, pred_len, 1) 真实标签（可选）
            method: 选择方法 ['latest', 'average', 'weighted', 'ensemble']
        
        Returns:
            final_predictions: 最终预测结果
            prediction_metadata: 预测元数据
        """
        n_samples, pred_len, _ = predictions.shape
        mappings = self.extract_date_mappings(n_samples + self.seq_len + self.pred_len - 1)
        
        # 创建日期到预测的映射
        date_predictions = {}
        date_targets = {}
        
        for mapping in mappings:
            sample_id = mapping['sample_id']
            pred_dates = mapping['prediction_dates']
            
            for i, date in enumerate(pred_dates):
                if date not in date_predictions:
                    date_predictions[date] = []
                    date_targets[date] = []
                
                date_predictions[date].append({
                    'sample_id': sample_id,
                    'prediction': predictions[sample_id, i, 0],
                    'prediction_day': i,  # 在预测序列中的位置
                    'input_end': mapping['input_range'][1]  # 输入序列结束位置
                })
                
                if targets is not None:
                    date_targets[date].append(targets[sample_id, i, 0])
        
        # 为每个日期选择最终预测
        final_predictions = {}
        prediction_metadata = {}
        
        for date in sorted(date_predictions.keys()):
            preds = date_predictions[date]
            
            if method == 'latest':
                # 选择最新的预测（输入序列结束位置最晚的）
                final_pred = max(preds, key=lambda x: x['input_end'])
                final_value = final_pred['prediction']
                metadata = {
                    'method': 'latest',
                    'selected_sample_id': final_pred['sample_id'],
                    'input_end': final_pred['input_end'],
                    'n_overlapping_predictions': len(preds),
                    'all_predictions': [p['prediction'] for p in preds]
                }
            
            elif method == 'average':
                # 选择所有预测的平均值
                all_preds = [p['prediction'] for p in preds]
                final_value = np.mean(all_preds)
                metadata = {
                    'method': 'average',
                    'n_overlapping_predictions': len(preds),
                    'all_predictions': all_preds,
                    'std': np.std(all_preds)
                }
            
            elif method == 'weighted':
                # 加权平均，越新的预测权重越高
                weights = []
                values = []
                for p in preds:
                    # 权重基于输入序列结束位置（越晚权重越高）
                    weight = p['input_end'] / max([pp['input_end'] for pp in preds])
                    weights.append(weight)
                    values.append(p['prediction'])
                
                weights = np.array(weights)
                weights = weights / weights.sum()  # 归一化
                final_value = np.average(values, weights=weights)
                metadata = {
                    'method': 'weighted',
                    'n_overlapping_predictions': len(preds),
                    'all_predictions': values,
                    'weights': weights.tolist(),
                    'weighted_std': np.sqrt(np.average((np.array(values) - final_value)**2, weights=weights))
                }
            
            elif method == 'ensemble':
                # 集成方法：结合多种策略
                latest_pred = max(preds, key=lambda x: x['input_end'])['prediction']
                avg_pred = np.mean([p['prediction'] for p in preds])
                
                # 如果预测差异很大，倾向于选择最新的
                pred_std = np.std([p['prediction'] for p in preds])
                if pred_std > 0.1:  # 阈值可调
                    final_value = latest_pred
                    selected_method = 'latest_due_to_high_variance'
                else:
                    final_value = avg_pred
                    selected_method = 'average_due_to_low_variance'
                
                metadata = {
                    'method': 'ensemble',
                    'selected_method': selected_method,
                    'n_overlapping_predictions': len(preds),
                    'all_predictions': [p['prediction'] for p in preds],
                    'latest_prediction': latest_pred,
                    'average_prediction': avg_pred,
                    'prediction_std': pred_std
                }
            
            final_predictions[date] = final_value
            prediction_metadata[date] = metadata
        
        return final_predictions, prediction_metadata
    
    def analyze_prediction_consistency(self, predictions, targets=None):
        """分析预测一致性"""
        n_samples, pred_len, _ = predictions.shape
        mappings = self.extract_date_mappings(n_samples + self.seq_len + self.pred_len - 1)
        
        consistency_analysis = {}
        
        for mapping in mappings:
            sample_id = mapping['sample_id']
            pred_dates = mapping['prediction_dates']
            
            for i, date in enumerate(pred_dates):
                if date not in consistency_analysis:
                    consistency_analysis[date] = {
                        'predictions': [],
                        'sample_ids': [],
                        'prediction_days': [],
                        'input_ends': []
                    }
                
                consistency_analysis[date]['predictions'].append(predictions[sample_id, i, 0])
                consistency_analysis[date]['sample_ids'].append(sample_id)
                consistency_analysis[date]['prediction_days'].append(i)
                consistency_analysis[date]['input_ends'].append(mapping['input_range'][1])
        
        # 计算一致性指标
        for date in consistency_analysis:
            preds = consistency_analysis[date]['predictions']
            consistency_analysis[date]['mean'] = np.mean(preds)
            consistency_analysis[date]['std'] = np.std(preds)
            consistency_analysis[date]['range'] = np.max(preds) - np.min(preds)
            consistency_analysis[date]['cv'] = np.std(preds) / np.mean(preds) if np.mean(preds) != 0 else 0
        
        return consistency_analysis

def load_and_process_predictions(folder_path, method='latest'):
    """加载并处理预测结果"""
    # 加载预测数据
    generated_file = os.path.join(folder_path, "generated_nsample15_guide0.8.npy")
    target_file = os.path.join(folder_path, "target_15_guide0.8.npy")
    
    if not os.path.exists(generated_file):
        print(f"预测文件不存在: {generated_file}")
        return None
    
    # 加载数据
    generated = np.load(generated_file)  # (batch_size, n_samples, seq_len+pred_len, 1)
    targets = np.load(target_file) if os.path.exists(target_file) else None
    
    # 取中位数作为预测结果
    predictions = np.median(generated, axis=1)  # (batch_size, seq_len+pred_len, 1)
    
    # 只取预测期
    seq_len = 36
    pred_len = 6
    future_predictions = predictions[:, seq_len:, 0]  # (batch_size, pred_len)
    future_targets = targets[:, seq_len:, 0] if targets is not None else None
    
    # 处理重叠预测
    handler = OverlappingPredictionHandler(seq_len, pred_len)
    
    # 分析一致性
    consistency = handler.analyze_prediction_consistency(future_predictions.reshape(-1, pred_len, 1))
    
    # 解决重叠预测
    final_preds, metadata = handler.resolve_overlapping_predictions(
        future_predictions.reshape(-1, pred_len, 1),
        future_targets.reshape(-1, pred_len, 1) if future_targets is not None else None,
        method=method
    )
    
    return final_preds, metadata, consistency

def create_final_prediction_report(folder_path, methods=['latest', 'average', 'weighted', 'ensemble']):
    """创建最终预测报告"""
    results = {}
    
    for method in methods:
        print(f"\n=== 处理方法: {method} ===")
        result = load_and_process_predictions(folder_path, method)
        if result:
            final_preds, metadata, consistency = result
            results[method] = {
                'predictions': final_preds,
                'metadata': metadata,
                'consistency': consistency
            }
            
            # 打印统计信息
            pred_values = list(final_preds.values())
            print(f"最终预测数量: {len(pred_values)}")
            print(f"预测均值: {np.mean(pred_values):.4f}")
            print(f"预测标准差: {np.std(pred_values):.4f}")
            
            # 分析重叠情况
            overlap_counts = [meta['n_overlapping_predictions'] for meta in metadata.values()]
            print(f"平均重叠预测数: {np.mean(overlap_counts):.2f}")
            print(f"最大重叠预测数: {np.max(overlap_counts)}")
    
    # 保存结果
    output_file = os.path.join(folder_path, "final_predictions_analysis.json")
    import json
    
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    import glob
    
    # 查找最新的结果文件夹
    result_folders = glob.glob("save/forecasting_copper_*")
    if not result_folders:
        print("未找到预测结果文件夹")
    else:
        latest_folder = max(result_folders, key=os.path.getctime)
        print(f"处理文件夹: {latest_folder}")
        
        # 创建最终预测报告
        results = create_final_prediction_report(latest_folder)
        
        # 显示不同方法的比较
        print("\n=== 方法比较 ===")
        for method, data in results.items():
            preds = data['predictions']
            values = list(preds.values())
            print(f"{method:10s}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}")
