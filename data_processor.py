import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataProcessor:
    def __init__(self, tokenizer_name: str = Config.MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = Config.MAX_SEQ_LENGTH
        self.sequence_labels = Config.SEQUENCE_LABELS
        self.group_labels = Config.GROUP_LABELS
        self.hate_labels = Config.HATE_LABELS
        
        # 创建标签到ID的映射
        self.seq_label2id = {label: i for i, label in enumerate(self.sequence_labels)}
        self.group_label2id = {label: i for i, label in enumerate(self.group_labels)}
        self.hate_label2id = {label: i for i, label in enumerate(self.hate_labels)}
        
        # 创建ID到标签的映射
        self.id2seq_label = {i: label for label, i in self.seq_label2id.items()}
        self.id2group_label = {i: label for label, i in self.group_label2id.items()}
        self.id2hate_label = {i: label for label, i in self.hate_label2id.items()}
    
    def load_data(self, file_path: str, is_test: bool = False) -> List[Dict]:
        """加载JSON数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def parse_output_string(self, output_string: str) -> List[Dict]:
        """解析训练数据中的output字符串为结构化数据"""
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
    
    def create_sequence_labels(self, text: str, quadruples: List[Dict]) -> List[str]:
        """基于四元组信息创建序列标注标签"""
        labels = ['O'] * len(text)
        
        for quad in quadruples:
            target = quad['target']
            argument = quad['argument']
            
            # 查找target在文本中的位置
            target_start = text.find(target)
            if target_start != -1 and target != 'NULL':
                labels[target_start] = 'B-TARGET'
                for i in range(target_start + 1, target_start + len(target)):
                    if i < len(labels):
                        labels[i] = 'I-TARGET'
            
            # 查找argument在文本中的位置
            arg_start = text.find(argument)
            if arg_start != -1:
                # 避免与target重叠
                if labels[arg_start] == 'O':
                    labels[arg_start] = 'B-ARG'
                    for i in range(arg_start + 1, arg_start + len(argument)):
                        if i < len(labels) and labels[i] == 'O':
                            labels[i] = 'I-ARG'
        
        return labels
    
    def tokenize_and_align_labels(self, text: str, labels: List[str] = None) -> Dict:
        """分词并对齐标签"""
        # 分词
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 获取词到字符的映射
        offset_mapping = tokenized['offset_mapping'][0].numpy()
        
        if labels is not None:
            # 对齐标签
            aligned_labels = []
            for start, end in offset_mapping:
                if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
                    aligned_labels.append('O')
                else:
                    # 取第一个字符的标签
                    if start < len(labels):
                        aligned_labels.append(labels[start])
                    else:
                        aligned_labels.append('O')
            
            # 转换为ID
            label_ids = [self.seq_label2id.get(label, 0) for label in aligned_labels]
            tokenized['labels'] = torch.tensor(label_ids, dtype=torch.long)
        
        # 移除offset_mapping
        del tokenized['offset_mapping']
        
        return tokenized
    
    def process_sample(self, sample: Dict, is_test: bool = False) -> Dict:
        """处理单个样本"""
        text = sample['content']
        processed = {
            'id': sample['id'],
            'text': text
        }
        
        if not is_test and 'output' in sample:
            # 解析四元组
            quadruples = self.parse_output_string(sample['output'])
            processed['quadruples'] = quadruples
            
            # 创建序列标签
            seq_labels = self.create_sequence_labels(text, quadruples)
            
            # 创建分类标签（取第一个四元组的标签作为全局标签）
            if quadruples:
                processed['group_label'] = quadruples[0]['group']
                processed['hate_label'] = quadruples[0]['hate']
            else:
                processed['group_label'] = 'non-hate'
                processed['hate_label'] = 'non-hate'
            
            # 分词和标签对齐
            tokenized = self.tokenize_and_align_labels(text, seq_labels)
            processed.update(tokenized)
            
            # 添加分类标签ID
            processed['group_label_id'] = self.group_label2id.get(processed['group_label'], 0)
            processed['hate_label_id'] = self.hate_label2id.get(processed['hate_label'], 0)
        else:
            # 测试数据只进行分词
            tokenized = self.tokenize_and_align_labels(text)
            processed.update(tokenized)
        
        return processed

class HateSpeechDataset(Dataset):
    def __init__(self, data: List[Dict], processor: HateSpeechDataProcessor, is_test: bool = False):
        self.data = data
        self.processor = processor
        self.is_test = is_test
        self.processed_data = []
        
        # 预处理所有数据
        for sample in data:
            processed = processor.process_sample(sample, is_test)
            self.processed_data.append(processed)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def collate_fn(batch):
    """自定义batch整理函数"""
    # 提取各个字段
    input_ids = torch.stack([item['input_ids'].squeeze() for item in batch])
    attention_mask = torch.stack([item['attention_mask'].squeeze() for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'ids': [item['id'] for item in batch],
        'texts': [item['text'] for item in batch]
    }
    
    # 如果是训练数据，添加标签
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        group_labels = torch.tensor([item['group_label_id'] for item in batch], dtype=torch.long)
        hate_labels = torch.tensor([item['hate_label_id'] for item in batch], dtype=torch.long)
        
        result.update({
            'labels': labels,
            'group_labels': group_labels,
            'hate_labels': hate_labels
        })
    
    return result

def create_dataloader(data: List[Dict], processor: HateSpeechDataProcessor, 
                     batch_size: int = Config.BATCH_SIZE, 
                     is_test: bool = False, shuffle: bool = True) -> DataLoader:
    """创建数据加载器"""
    dataset = HateSpeechDataset(data, processor, is_test)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle and not is_test,
        collate_fn=collate_fn,
        num_workers=0
    ) 