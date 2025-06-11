import os
import torch
import logging
from tqdm import tqdm

from config import Config
from data_processor import HateSpeechDataProcessor, create_dataloader
from model import MultiTaskHateSpeechModel
from utils import set_seed, load_model_checkpoint, QuadrupleExtractor, format_output, create_submission_file

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
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
    
    def predict_batch(self, batch):
        """对一个batch进行预测"""
        with torch.no_grad():
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            texts = batch['texts']
            
            # 获取模型预测结果
            predictions = self.model.predict(input_ids, attention_mask)
            
            # 后处理预测结果
            batch_results = []
            for i in range(len(texts)):
                text = texts[i]
                seq_pred = predictions['sequence_predictions'][i].cpu().numpy()
                group_pred = predictions['group_predictions'][i].item()
                hate_pred = predictions['hate_predictions'][i].item()
                relation_scores = predictions['relation_scores'][i].cpu()
                
                # 提取四元组
                quadruples = self.extractor.post_process_predictions(
                    text, seq_pred, group_pred, hate_pred, relation_scores
                )
                
                # 格式化输出
                formatted_output = format_output(quadruples)
                batch_results.append(formatted_output)
            
            return batch_results
    
    def predict_test_set(self, test_file: str, output_file: str):
        """对测试集进行预测"""
        logger.info(f"Loading test data from {test_file}")
        test_data = self.processor.load_data(test_file, is_test=True)
        
        # 创建数据加载器
        test_loader = create_dataloader(
            test_data, self.processor, 
            batch_size=self.config.BATCH_SIZE, 
            is_test=True, shuffle=False
        )
        
        logger.info(f"Predicting {len(test_data)} samples...")
        
        all_predictions = []
        sample_ids = []
        
        for batch in tqdm(test_loader, desc="Predicting"):
            # 获取预测结果
            batch_predictions = self.predict_batch(batch)
            all_predictions.extend(batch_predictions)
            sample_ids.extend(batch['ids'])
        
        # 按照ID排序确保顺序正确
        id_pred_pairs = list(zip(sample_ids, all_predictions))
        id_pred_pairs.sort(key=lambda x: x[0])
        sorted_predictions = [pred for _, pred in id_pred_pairs]
        
        # 保存预测结果
        create_submission_file(sorted_predictions, output_file)
        logger.info(f"Predictions saved to {output_file}")
        
        return sorted_predictions
    
    def predict_single_text(self, text: str) -> str:
        """对单个文本进行预测"""
        # 准备输入数据
        sample = {'id': 0, 'content': text}
        processed = self.processor.process_sample(sample, is_test=True)
        
        # 转换为batch格式
        batch = {
            'input_ids': processed['input_ids'].unsqueeze(0),
            'attention_mask': processed['attention_mask'].unsqueeze(0),
            'texts': [text]
        }
        
        # 预测
        predictions = self.predict_batch(batch)
        return predictions[0]

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="仇恨言论检测预测脚本")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--test_file", type=str, default=Config.TEST1_FILE,
                       help="测试文件路径")
    parser.add_argument("--output_file", type=str, default="predictions.txt",
                       help="输出文件路径")
    parser.add_argument("--text", type=str, default=None,
                       help="单个文本预测")
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = Predictor(args.model_path)
    
    if args.text:
        # 单个文本预测
        result = predictor.predict_single_text(args.text)
        print(f"输入文本: {args.text}")
        print(f"预测结果: {result}")
    else:
        # 测试集预测
        predictor.predict_test_set(args.test_file, args.output_file)
        print(f"预测完成，结果保存在 {args.output_file}")

if __name__ == "__main__":
    main()

# 使用示例:
# python predict.py --model_path saved_models/best_model.pt --test_file test1.json --output_file test1_predictions.txt
# python predict.py --model_path saved_models/best_model.pt --text "这是一个测试文本" 