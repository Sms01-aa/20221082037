import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss
from typing import Dict, Optional, Tuple, List
from config import Config

class MultiTaskHateSpeechModel(nn.Module):
    """多任务仇恨言论检测模型"""
    
    def __init__(self, model_name: str = Config.MODEL_NAME):
        super().__init__()
        
        # 加载预训练模型配置和模型
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 模型参数
        self.hidden_size = self.config.hidden_size
        self.num_seq_labels = len(Config.SEQUENCE_LABELS)
        self.num_group_labels = len(Config.GROUP_LABELS) 
        self.num_hate_labels = len(Config.HATE_LABELS)
        
        # Dropout层
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        # 序列标注头（Token级别）
        self.sequence_classifier = nn.Linear(self.hidden_size, self.num_seq_labels)
        
        # 分类头（句子级别）
        self.group_classifier = nn.Linear(self.hidden_size, self.num_group_labels)
        self.hate_classifier = nn.Linear(self.hidden_size, self.num_hate_labels)
        
        # 简化的关系抽取头（用于连接Target和Argument）
        self.relation_head = nn.Linear(self.hidden_size * 2, 1)
        
        # 损失函数
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)  # 处理类别不平衡
        
        # 损失权重
        self.seq_loss_weight = 1.0
        self.group_loss_weight = 1.0
        self.hate_loss_weight = 2.0  # 仇恨检测更重要
        self.relation_loss_weight = 0.5
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                group_labels: Optional[torch.Tensor] = None,
                hate_labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        
        # 获取BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output        # [batch_size, hidden_size]
        
        # 应用dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # 序列标注预测
        sequence_logits = self.sequence_classifier(sequence_output)
        
        # 分类预测
        group_logits = self.group_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)
        
        # 关系抽取预测（简化版本，使用平均池化）
        pooled_sequence = torch.mean(sequence_output, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        pooled_expanded = pooled_sequence.expand(-1, sequence_output.size(1), -1)  # [batch_size, seq_len, hidden_size]
        relation_input = torch.cat([sequence_output, pooled_expanded], dim=-1)  # [batch_size, seq_len, hidden_size*2]
        relation_scores = self.relation_head(relation_input).squeeze(-1)  # [batch_size, seq_len]
        
        # 组织输出
        outputs_dict = {
            'sequence_logits': sequence_logits,
            'group_logits': group_logits,
            'hate_logits': hate_logits,
            'relation_scores': relation_scores
        }
        
        # 计算损失
        if labels is not None:
            total_loss = 0
            
            # 序列标注损失
            seq_loss = self.loss_fct(
                sequence_logits.view(-1, self.num_seq_labels),
                labels.view(-1)
            )
            total_loss += self.seq_loss_weight * seq_loss
            outputs_dict['seq_loss'] = seq_loss
            
            # 群体分类损失
            if group_labels is not None:
                group_loss = self.focal_loss(group_logits, group_labels)
                total_loss += self.group_loss_weight * group_loss
                outputs_dict['group_loss'] = group_loss
            
            # 仇恨检测损失
            if hate_labels is not None:
                hate_loss = self.focal_loss(hate_logits, hate_labels)
                total_loss += self.hate_loss_weight * hate_loss
                outputs_dict['hate_loss'] = hate_loss
            
            outputs_dict['loss'] = total_loss
        
        return outputs_dict
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """预测函数"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # 获取预测结果
            seq_preds = torch.argmax(outputs['sequence_logits'], dim=-1)
            group_preds = torch.argmax(outputs['group_logits'], dim=-1)
            hate_preds = torch.argmax(outputs['hate_logits'], dim=-1)
            
            return {
                'sequence_predictions': seq_preds,
                'group_predictions': group_preds, 
                'hate_predictions': hate_preds,
                'relation_scores': outputs['relation_scores']
            }

# BiaffineAttention类已移除，使用更简单的关系抽取方法

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 