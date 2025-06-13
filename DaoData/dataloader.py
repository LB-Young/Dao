import json
import torch
from torch.utils.data import DataLoader

class BaseDataGenerator:
    def __init__(self, file_path, tokenizer, max_length):
        # 基础数据加载器的初始化方法
        self.file_path = file_path  # 存储文件路径
        self.tokenizer = tokenizer  # 存储分词器
        self.max_length = max_length  # 存储最大长度
        self.load()

    def __len__(self):
        return len(self.data)
    
    def load(self):
        self.datas = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.datas.append(obj)
        return


class PretrainDataGenerator(BaseDataGenerator):
    def __init__(self, file_path, tokenizer, max_length):
        # 预训练数据加载器的初始化方法
        # 调用父类的初始化方法，传递所有必需的参数
        super().__init__(file_path, tokenizer, max_length)

    def __getitem__(self, index):
        sample = self.datas[index]

        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


class SFTDataGenerator(BaseDataGenerator):
    def __init__(self, file_path, tokenizer, max_length):
        # 预训练数据加载器的初始化方法
        # 调用父类的初始化方法，传递所有必需的参数
        super().__init__(file_path, tokenizer, max_length)

    def __getitem__(self, index):
        pass

class DPODataGenerator(BaseDataGenerator):
    def __init__(self, file_path, tokenizer, max_length):
        # 预训练数据加载器的初始化方法
        # 调用父类的初始化方法，传递所有必需的参数
        super().__init__(file_path, tokenizer, max_length)

    def __getitem__(self, index):
        pass

class PPODataGenerator(BaseDataGenerator):
    def __init__(self, file_path, tokenizer, max_length):
        # 预训练数据加载器的初始化方法
        # 调用父类的初始化方法，传递所有必需的参数
        super().__init__(file_path, tokenizer, max_length)

    def __getitem__(self, index):
        pass

class GRPODataGenerator(BaseDataGenerator):
    def __init__(self, file_path, tokenizer, max_length):
        # 预训练数据加载器的初始化方法
        # 调用父类的初始化方法，传递所有必需的参数
        super().__init__(file_path, tokenizer, max_length)

    def __getitem__(self, index):
        pass

class DAPODataGenerator(BaseDataGenerator):
    def __init__(self, file_path, tokenizer, max_length):
        # 预训练数据加载器的初始化方法
        # 调用父类的初始化方法，传递所有必需的参数
        super().__init__(file_path, tokenizer, max_length)

    def __getitem__(self, index):
        pass


class DataGenerator:
    def __init__(self, file_path, tokenizer, max_length=1024, data_type="pretrain"):
        # 通用数据加载器的初始化方法
        if data_type == "pretrain":
            self.data_generator = PretrainDataGenerator(file_path=file_path, tokenizer=tokenizer, max_length=max_length)
        elif data_type == "sft":
            self.data_generator = SFTDataGenerator(file_path=file_path, tokenizer=tokenizer, max_length=max_length)
        elif data_type == "dpo":
            self.data_generator = DPODataGenerator(file_path=file_path, tokenizer=tokenizer, max_length=max_length)
        elif data_type == "ppo":
            self.data_generator = PPODataGenerator(file_path=file_path, tokenizer=tokenizer, max_length=max_length)
        elif data_type == "grpo":
            self.data_generator = GRPODataGenerator(file_path=file_path, tokenizer=tokenizer, max_length=max_length)
        elif data_type == "dapo":
            self.data_generator = DAPODataGenerator(file_path=file_path, tokenizer=tokenizer, max_length=max_length)
        else:
            raise ValueError(f"Invalid data type: {data_type}") 
    
    def __len__(self):
        return len(self.data_loader)
    
    def __getitem__(self, index):
        return self.data_loader[index]  


def data_loader(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config["tokenizer"], max_length=config["max_length"], data_type=config["data_type"])
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # print(len(dl))
    return dl

def test_datagenerator():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/Users/liubaoyang/Documents/YoungL/project/github_projects/minimind/model")
    data_generator = DataGenerator(file_path="/Users/liubaoyang/Documents/YoungL/project/github_projects/minimind/dataset/pretrain_hq.jsonl", tokenizer=tokenizer, max_length=1024, data_type="pretrain")
    for data in data_generator:
        print(data)
