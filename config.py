import os

class Config:
    # 数据相关配置
    TRAIN_FILE = "train.json"
    TEST1_FILE = "test1.json" 
    TEST2_FILE = "test2.json"
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_DIR = "saved_models"
    
    # 模型相关配置
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 中文RoBERTa模型
    MAX_SEQ_LENGTH = 256  # 减小序列长度以节省内存
    HIDDEN_SIZE = 768
    NUM_LABELS_GROUP = 6  # non-hate, Racism, Sexism, Region, LGBTQ, others
    NUM_LABELS_HATE = 2   # hate, non-hate
    
    # 训练相关配置
    BATCH_SIZE = 8  # 减小批次大小以节省内存
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    
    # 序列标注标签
    SEQUENCE_LABELS = [
        'O',        # 其他
        'B-TARGET', # Target开始
        'I-TARGET', # Target内部
        'B-ARG',    # Argument开始
        'I-ARG',    # Argument内部
        'SEP',      # 分隔符
        'END'       # 结束符
    ]
    
    # 目标群体标签
    GROUP_LABELS = ['non-hate', 'Racism', 'Sexism', 'Region', 'LGBTQ', 'others']
    
    # 仇恨标签
    HATE_LABELS = ['non-hate', 'hate']
    
    # 设备配置
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    SEED = 42 