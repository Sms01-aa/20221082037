import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import re
from config import Config

def set_seed(seed: int = Config.SEED):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class QuadrupleExtractor:
    """从模型预测结果中提取四元组"""
    
    def __init__(self, processor):
        self.processor = processor
        self.seq_labels = Config.SEQUENCE_LABELS
        self.group_labels = Config.GROUP_LABELS
        self.hate_labels = Config.HATE_LABELS
    
    def extract_entities(self, tokens: List[str], labels: List[str], label_type: str) -> List[Dict]:
        """提取实体"""
        entities = []
        start_label = f'B-{label_type}'
        inside_label = f'I-{label_type}'
        
        start_idx = None
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label == start_label:
                if start_idx is not None:
                    # 结束前一个实体
                    entities.append({
                        'text': ''.join(tokens[start_idx:i]),
                        'start': start_idx,
                        'end': i
                    })
                start_idx = i
            elif label != inside_label and start_idx is not None:
                # 结束当前实体
                entities.append({
                    'text': ''.join(tokens[start_idx:i]),
                    'start': start_idx,
                    'end': i
                })
                start_idx = None
        
        # 处理最后一个实体
        if start_idx is not None:
            entities.append({
                'text': ''.join(tokens[start_idx:]),
                'start': start_idx,
                'end': len(tokens)
            })
        
        return entities
    
    def match_target_argument(self, targets: List[Dict], arguments: List[Dict], 
                            relation_scores: torch.Tensor) -> List[Tuple[Dict, Dict]]:
        """基于关系得分匹配Target和Argument"""
        pairs = []
        used_args = set()
        
        for target in targets:
            best_score = -float('inf')
            best_arg = None
            
            for arg in arguments:
                if arg['start'] in used_args:
                    continue
                
                # 计算关系得分（简化版本）
                if target['start'] < len(relation_scores) and arg['start'] < len(relation_scores):
                    score = relation_scores[target['start']].item() + relation_scores[arg['start']].item()
                    if score > best_score:
                        best_score = score
                        best_arg = arg
            
            if best_arg is not None:
                pairs.append((target, best_arg))
                used_args.add(best_arg['start'])
        
        return pairs
    
    def post_process_predictions(self, text: str, seq_preds: List[int], 
                               group_preds: int, hate_preds: int,
                               relation_scores: torch.Tensor) -> List[Dict]:
        """后处理预测结果，提取四元组"""
        # 分词
        tokens = list(text)  # 字符级分词
        
        # 转换序列标签
        seq_labels = [self.processor.id2seq_label.get(pred, 'O') for pred in seq_preds[:len(tokens)]]
        
        # 提取实体
        targets = self.extract_entities(tokens, seq_labels, 'TARGET')
        arguments = self.extract_entities(tokens, seq_labels, 'ARG')
        
        # 匹配Target-Argument对
        pairs = self.match_target_argument(targets, arguments, relation_scores)
        
        # 构建四元组
        quadruples = []
        group_label = self.processor.id2group_label.get(group_preds, 'non-hate')
        hate_label = self.processor.id2hate_label.get(hate_preds, 'non-hate')
        
        for target, argument in pairs:
            quadruples.append({
                'target': target['text'],
                'argument': argument['text'],
                'group': group_label,
                'hate': hate_label
            })
        
        # 如果没有找到有效的四元组，创建一个默认的
        if not quadruples and (targets or arguments):
            target_text = targets[0]['text'] if targets else 'NULL'
            arg_text = arguments[0]['text'] if arguments else text[:min(10, len(text))]
            
            quadruples.append({
                'target': target_text,
                'argument': arg_text, 
                'group': group_label,
                'hate': hate_label
            })
        
        return quadruples

def format_output(quadruples: List[Dict]) -> str:
    """将四元组格式化为提交格式"""
    if not quadruples:
        return "NULL | NULL | non-hate | non-hate [END]"
    
    formatted_parts = []
    for quad in quadruples:
        part = f"{quad['target']} | {quad['argument']} | {quad['group']} | {quad['hate']}"
        formatted_parts.append(part)
    
    return " [SEP] ".join(formatted_parts) + " [END]"

class EvaluationMetrics:
    """评估指标计算"""
    
    @staticmethod
    def compute_string_similarity(str1: str, str2: str) -> float:
        """计算字符串相似度"""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1, str2).ratio()
    
    @staticmethod
    def is_soft_match(pred_quad: Dict, gold_quad: Dict, threshold: float = 0.5) -> bool:
        """软匹配判断"""
        target_sim = EvaluationMetrics.compute_string_similarity(
            pred_quad['target'], gold_quad['target']
        )
        arg_sim = EvaluationMetrics.compute_string_similarity(
            pred_quad['argument'], gold_quad['argument']
        )
        
        return (target_sim >= threshold and arg_sim >= threshold and
                pred_quad['group'] == gold_quad['group'] and
                pred_quad['hate'] == gold_quad['hate'])
    
    @staticmethod
    def is_hard_match(pred_quad: Dict, gold_quad: Dict) -> bool:
        """硬匹配判断"""
        return (pred_quad['target'] == gold_quad['target'] and
                pred_quad['argument'] == gold_quad['argument'] and
                pred_quad['group'] == gold_quad['group'] and
                pred_quad['hate'] == gold_quad['hate'])
    
    @staticmethod
    def compute_f1_score(predictions: List[List[Dict]], 
                        gold_standards: List[List[Dict]], 
                        match_type: str = 'hard') -> Dict[str, float]:
        """计算F1分数"""
        match_func = EvaluationMetrics.is_hard_match if match_type == 'hard' else EvaluationMetrics.is_soft_match
        
        total_pred = 0
        total_gold = 0
        total_correct = 0
        
        for pred_quads, gold_quads in zip(predictions, gold_standards):
            total_pred += len(pred_quads)
            total_gold += len(gold_quads)
            
            # 计算匹配的四元组数量
            matched_gold = set()
            for pred_quad in pred_quads:
                for i, gold_quad in enumerate(gold_quads):
                    if i not in matched_gold and match_func(pred_quad, gold_quad):
                        total_correct += 1
                        matched_gold.add(i)
                        break
        
        # 计算精确率、召回率和F1
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_gold if total_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_pred': total_pred,
            'total_gold': total_gold,
            'total_correct': total_correct
        }

def parse_output_string(output_string: str) -> List[Dict]:
    """解析输出字符串为四元组列表"""
    quadruples = []
    
    # 按[SEP]分割多个四元组
    parts = re.split(r'\s*\[SEP\]\s*', output_string.strip())
    
    for part in parts:
        # 移除[END]标记
        part = re.sub(r'\s*\[END\]\s*$', '', part)
        if not part.strip():
            continue
            
        # 按|分割四个元素
        elements = [elem.strip() for elem in part.split('|')]
        if len(elements) == 4:
            target, argument, group, hate = elements
            quadruples.append({
                'target': target,
                'argument': argument,
                'group': group,
                'hate': hate
            })
    
    return quadruples

def clean_text(text: str) -> str:
    """清理文本"""
    # 移除多余的空白符
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 处理特殊字符
    text = text.replace('\u200b', '')  # 零宽空格
    text = text.replace('\ufeff', '')  # BOM
    
    return text

def create_submission_file(predictions: List[str], output_file: str):
    """创建提交文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')

def load_model_checkpoint(model, checkpoint_path: str):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def save_model_checkpoint(model, optimizer, epoch: int, loss: float, save_path: str):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)

class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_loss: float) -> bool:
        """
        判断是否应该早停
        
        Args:
            val_loss: 验证损失
            
        Returns:
            bool: 是否应该早停
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            
        return False 