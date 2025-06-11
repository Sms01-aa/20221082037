import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

from config import Config
from data_processor import HateSpeechDataProcessor, create_dataloader
from model import MultiTaskHateSpeechModel
from utils import set_seed, save_model_checkpoint, EvaluationMetrics, parse_output_string

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config=Config):
        self.config = config
        
        # 设置随机种子
        set_seed(config.SEED)
        
        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        
        # 初始化数据处理器
        self.processor = HateSpeechDataProcessor(config.MODEL_NAME)
        
        # 初始化模型
        self.model = MultiTaskHateSpeechModel(config.MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self):
        """准备训练数据"""
        logger.info("Loading training data...")
        train_data = self.processor.load_data(self.config.TRAIN_FILE)
        
        # 划分训练集和验证集
        train_data, val_data = train_test_split(
            train_data, test_size=0.1, random_state=self.config.SEED
        )
        
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        
        # 创建数据加载器
        self.train_loader = create_dataloader(
            train_data, self.processor, self.config.BATCH_SIZE, shuffle=True
        )
        self.val_loader = create_dataloader(
            val_data, self.processor, self.config.BATCH_SIZE, shuffle=False
        )
        
        return len(train_data)
    
    def setup_optimizer_and_scheduler(self, num_training_samples):
        """设置优化器和学习率调度器"""
        # 计算总的训练步数
        num_training_steps = (num_training_samples // self.config.BATCH_SIZE) * self.config.NUM_EPOCHS
        num_warmup_steps = int(num_training_steps * self.config.WARMUP_RATIO)
        
        # 设置优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 设置学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_seq_loss = 0
        total_group_loss = 0
        total_hate_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            group_labels = batch['group_labels'].to(self.device)
            hate_labels = batch['hate_labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                group_labels=group_labels,
                hate_labels=hate_labels
            )
            
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 记录损失
            total_loss += loss.item()
            if 'seq_loss' in outputs:
                total_seq_loss += outputs['seq_loss'].item()
            if 'group_loss' in outputs:
                total_group_loss += outputs['group_loss'].item()
            if 'hate_loss' in outputs:
                total_hate_loss += outputs['hate_loss'].item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 定期保存模型
            if (step + 1) % self.config.SAVE_STEPS == 0:
                save_path = os.path.join(
                    self.config.MODEL_SAVE_DIR, 
                    f"checkpoint-epoch-{epoch}-step-{step+1}.pt"
                )
                save_model_checkpoint(self.model, self.optimizer, epoch, loss.item(), save_path)
                logger.info(f"Model saved at step {step+1}")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_seq_loss = total_seq_loss / len(self.train_loader)
        avg_group_loss = total_group_loss / len(self.train_loader)
        avg_hate_loss = total_hate_loss / len(self.train_loader)
        
        return {
            'avg_loss': avg_loss,
            'avg_seq_loss': avg_seq_loss,
            'avg_group_loss': avg_group_loss,
            'avg_hate_loss': avg_hate_loss
        }
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        gold_standards = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                group_labels = batch['group_labels'].to(self.device)
                hate_labels = batch['hate_labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    group_labels=group_labels,
                    hate_labels=hate_labels
                )
                
                total_loss += outputs['loss'].item()
                
                # 获取预测结果
                preds = self.model.predict(input_ids, attention_mask)
                
                # 后处理预测结果
                for i in range(len(batch['texts'])):
                    text = batch['texts'][i]
                    seq_pred = preds['sequence_predictions'][i].cpu().numpy()
                    group_pred = preds['group_predictions'][i].item()
                    hate_pred = preds['hate_predictions'][i].item()
                    relation_scores = preds['relation_scores'][i].cpu()
                    
                    # 提取四元组
                    from utils import QuadrupleExtractor
                    extractor = QuadrupleExtractor(self.processor)
                    pred_quads = extractor.post_process_predictions(
                        text, seq_pred, group_pred, hate_pred, relation_scores
                    )
                    predictions.append(pred_quads)
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {'avg_loss': avg_loss}
    
    def train(self):
        """完整的训练流程"""
        logger.info("Starting training...")
        
        # 准备数据
        num_training_samples = self.prepare_data()
        
        # 设置优化器和调度器
        self.setup_optimizer_and_scheduler(num_training_samples)
        
        best_loss = float('inf')
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            logger.info(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
                       f"Seq Loss: {train_metrics['avg_seq_loss']:.4f}, "
                       f"Group Loss: {train_metrics['avg_group_loss']:.4f}, "
                       f"Hate Loss: {train_metrics['avg_hate_loss']:.4f}")
            
            # 评估
            if epoch % (self.config.EVAL_STEPS // len(self.train_loader) + 1) == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Eval - Loss: {eval_metrics['avg_loss']:.4f}")
                
                # 保存最佳模型
                if eval_metrics['avg_loss'] < best_loss:
                    best_loss = eval_metrics['avg_loss']
                    best_model_path = os.path.join(self.config.MODEL_SAVE_DIR, "best_model.pt")
                    save_model_checkpoint(
                        self.model, self.optimizer, epoch, 
                        eval_metrics['avg_loss'], best_model_path
                    )
                    logger.info(f"New best model saved with loss: {best_loss:.4f}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.config.MODEL_SAVE_DIR, "final_model.pt")
        save_model_checkpoint(
            self.model, self.optimizer, self.config.NUM_EPOCHS, 
            train_metrics['avg_loss'], final_model_path
        )
        logger.info("Training completed!")

def main():
    """主函数"""
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main() 