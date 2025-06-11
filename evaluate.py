import os
import torch
import logging
from tqdm import tqdm
import json

from config import Config
from data_processor import HateSpeechDataProcessor, create_dataloader
from model import MultiTaskHateSpeechModel
from utils import (set_seed, load_model_checkpoint, QuadrupleExtractor, 
                  EvaluationMetrics, parse_output_string)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model_path: str, config=Config):
        self.config = config
        self.model_path = model_path
        
        # 设置随机种子
        set_seed(config.SEED)
        
        # 初始化数据处理器
        self.processor = HateSpeechDataProcessor(config.MODEL_NAME)
        
        # 初始化模型
        self.model = MultiTaskHateSpeechModel(config.MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载训练好的模型
        logger.info(f"Loading model from {model_path}")
        load_model_checkpoint(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化四元组提取器
        self.extractor = QuadrupleExtractor(self.processor)
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def evaluate_on_dataset(self, data_file: str):
        """在数据集上进行评估"""
        logger.info(f"Loading data from {data_file}")
        data = self.processor.load_data(data_file)
        
        # 创建数据加载器
        data_loader = create_dataloader(
            data, self.processor, 
            batch_size=self.config.BATCH_SIZE, 
            is_test=False, shuffle=False
        )
        
        logger.info(f"Evaluating on {len(data)} samples...")
        
        predictions = []
        gold_standards = []
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            texts = batch['texts']
            
            # 获取模型预测结果
            with torch.no_grad():
                preds = self.model.predict(input_ids, attention_mask)
            
            # 后处理预测结果并获取标准答案
            for i in range(len(texts)):
                text = texts[i]
                sample_id = batch['ids'][i]
                
                # 提取预测的四元组
                seq_pred = preds['sequence_predictions'][i].cpu().numpy()
                group_pred = preds['group_predictions'][i].item()
                hate_pred = preds['hate_predictions'][i].item()
                relation_scores = preds['relation_scores'][i].cpu()
                
                pred_quads = self.extractor.post_process_predictions(
                    text, seq_pred, group_pred, hate_pred, relation_scores
                )
                predictions.append(pred_quads)
                
                # 获取标准答案
                original_sample = next(sample for sample in data if sample['id'] == sample_id)
                if 'output' in original_sample:
                    gold_quads = parse_output_string(original_sample['output'])
                    gold_standards.append(gold_quads)
                else:
                    # 如果没有标准答案，创建空列表
                    gold_standards.append([])
        
        return predictions, gold_standards
    
    def compute_metrics(self, predictions, gold_standards):
        """计算评估指标"""
        # 计算硬匹配F1
        hard_metrics = EvaluationMetrics.compute_f1_score(
            predictions, gold_standards, match_type='hard'
        )
        
        # 计算软匹配F1
        soft_metrics = EvaluationMetrics.compute_f1_score(
            predictions, gold_standards, match_type='soft'
        )
        
        # 计算平均F1
        avg_f1 = (hard_metrics['f1'] + soft_metrics['f1']) / 2
        
        results = {
            'hard_match': hard_metrics,
            'soft_match': soft_metrics,
            'average_f1': avg_f1
        }
        
        return results
    
    def detailed_analysis(self, predictions, gold_standards, save_file=None):
        """详细错误分析"""
        analysis_results = {
            'total_samples': len(predictions),
            'perfect_matches': 0,
            'partial_matches': 0,
            'no_matches': 0,
            'error_types': {
                'target_errors': 0,
                'argument_errors': 0,
                'group_errors': 0,
                'hate_errors': 0
            },
            'examples': {
                'perfect': [],
                'partial': [],
                'errors': []
            }
        }
        
        for i, (pred_quads, gold_quads) in enumerate(zip(predictions, gold_standards)):
            if not gold_quads:  # 跳过没有标准答案的样本
                continue
                
            # 检查完全匹配
            perfect_match = False
            if len(pred_quads) == len(gold_quads):
                matches = 0
                for pred_quad in pred_quads:
                    for gold_quad in gold_quads:
                        if EvaluationMetrics.is_hard_match(pred_quad, gold_quad):
                            matches += 1
                            break
                if matches == len(gold_quads):
                    perfect_match = True
            
            if perfect_match:
                analysis_results['perfect_matches'] += 1
                if len(analysis_results['examples']['perfect']) < 5:
                    analysis_results['examples']['perfect'].append({
                        'index': i,
                        'prediction': pred_quads,
                        'gold': gold_quads
                    })
            else:
                # 检查部分匹配
                partial_match = False
                for pred_quad in pred_quads:
                    for gold_quad in gold_quads:
                        if EvaluationMetrics.is_soft_match(pred_quad, gold_quad):
                            partial_match = True
                            break
                    if partial_match:
                        break
                
                if partial_match:
                    analysis_results['partial_matches'] += 1
                    if len(analysis_results['examples']['partial']) < 5:
                        analysis_results['examples']['partial'].append({
                            'index': i,
                            'prediction': pred_quads,
                            'gold': gold_quads
                        })
                else:
                    analysis_results['no_matches'] += 1
                    if len(analysis_results['examples']['errors']) < 5:
                        analysis_results['examples']['errors'].append({
                            'index': i,
                            'prediction': pred_quads,
                            'gold': gold_quads
                        })
                
                # 分析错误类型
                for pred_quad in pred_quads:
                    best_gold = None
                    best_similarity = 0
                    
                    for gold_quad in gold_quads:
                        # 计算整体相似度
                        target_sim = EvaluationMetrics.compute_string_similarity(
                            pred_quad['target'], gold_quad['target']
                        )
                        arg_sim = EvaluationMetrics.compute_string_similarity(
                            pred_quad['argument'], gold_quad['argument']
                        )
                        overall_sim = (target_sim + arg_sim) / 2
                        
                        if overall_sim > best_similarity:
                            best_similarity = overall_sim
                            best_gold = gold_quad
                    
                    if best_gold:
                        if pred_quad['target'] != best_gold['target']:
                            analysis_results['error_types']['target_errors'] += 1
                        if pred_quad['argument'] != best_gold['argument']:
                            analysis_results['error_types']['argument_errors'] += 1
                        if pred_quad['group'] != best_gold['group']:
                            analysis_results['error_types']['group_errors'] += 1
                        if pred_quad['hate'] != best_gold['hate']:
                            analysis_results['error_types']['hate_errors'] += 1
        
        if save_file:
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed analysis saved to {save_file}")
        
        return analysis_results
    
    def evaluate(self, data_file: str, save_analysis: bool = True):
        """完整的评估流程"""
        # 获取预测结果和标准答案
        predictions, gold_standards = self.evaluate_on_dataset(data_file)
        
        # 计算评估指标
        metrics = self.compute_metrics(predictions, gold_standards)
        
        # 打印结果
        logger.info("=== 评估结果 ===")
        logger.info(f"硬匹配 - P: {metrics['hard_match']['precision']:.4f}, "
                   f"R: {metrics['hard_match']['recall']:.4f}, "
                   f"F1: {metrics['hard_match']['f1']:.4f}")
        logger.info(f"软匹配 - P: {metrics['soft_match']['precision']:.4f}, "
                   f"R: {metrics['soft_match']['recall']:.4f}, "
                   f"F1: {metrics['soft_match']['f1']:.4f}")
        logger.info(f"平均F1: {metrics['average_f1']:.4f}")
        
        # 详细分析
        if save_analysis:
            analysis_file = os.path.join(self.config.OUTPUT_DIR, "error_analysis.json")
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            analysis = self.detailed_analysis(predictions, gold_standards, analysis_file)
            
            logger.info("=== 错误分析 ===")
            logger.info(f"完全匹配: {analysis['perfect_matches']}")
            logger.info(f"部分匹配: {analysis['partial_matches']}")
            logger.info(f"无匹配: {analysis['no_matches']}")
            logger.info(f"目标错误: {analysis['error_types']['target_errors']}")
            logger.info(f"论点错误: {analysis['error_types']['argument_errors']}")
            logger.info(f"群体错误: {analysis['error_types']['group_errors']}")
            logger.info(f"仇恨错误: {analysis['error_types']['hate_errors']}")
        
        return metrics

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="仇恨言论检测评估脚本")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--data_file", type=str, default=Config.TRAIN_FILE,
                       help="评估数据文件路径")
    parser.add_argument("--no_analysis", action="store_true",
                       help="不进行详细错误分析")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = Evaluator(args.model_path)
    
    # 进行评估
    metrics = evaluator.evaluate(args.data_file, save_analysis=not args.no_analysis)
    
    print("\n=== 最终结果 ===")
    print(f"硬匹配F1: {metrics['hard_match']['f1']:.4f}")
    print(f"软匹配F1: {metrics['soft_match']['f1']:.4f}")
    print(f"平均F1: {metrics['average_f1']:.4f}")

if __name__ == "__main__":
    main()

# 使用示例:
# python evaluate.py --model_path saved_models/best_model.pt --data_file train.json 