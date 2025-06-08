import torch
import torch.nn as nn

# 层归一化
class DaoLayerNorm(nn.Module):
    """
    RMSNorm
    """
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: h的最后一个维度即 batch * seq_len * dim中的dim
            eps: 防止分母为0
        """
        self.eps = 1e-6 # 防止分母为0
        self.gamma = nn.Parameter(torch.ones(dim)) # gamma为缩放因子，初始化为1

    def forward(self, h):
        """
        formula:
            h / h的均方根：
            h = h / sqrt(sum(h_i ** 2) / h_dim + self.eps)
            = h * rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        dim transform:
            h : batch * seq_len * dim
            h.pow(2) : batch * seq_len * dim    dim维度上每个元素平方
            h.pow(2).mean(-1, keepdim=True) : batch * seq_len * 1    dim维度上所有元素平方的均值
            rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) : batch * seq_len * 1    dim维度上每个元素平方的均值的平方根的倒数
            h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) : batch * seq_len * dim    dim维度上每个元素平方的均值的平方根的倒数乘以h，batch * seq_len * dim 对位乘 batch * seq_len * 1（batch * seq_len * 1中的1会自动broadcast） => batch * seq_len * dim

            self.gamma : dim
            self.gamma * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)) : batch * seq_len * dim（batch * seq_len * dim 对位乘 dim， 其中dim会自动broadcast为batch * seq_len * dim）
        """
        return self.gamma * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps))
      
class DaoMHA(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, attention_config):
        self.attention_config = attention_config
        pass
    
    def forward(self):
        pass

class DaoGQA(nn.Module):
    """
    GQA Attention
    """
    def __init__(self, attention_config):
        self.attention_config = attention_config
        pass

    def forward(self):
        pass

class DaoMQA(nn.Module):
    """
    MQA Attention
    """
    def __init__(self, attention_config):
        self.attention_config = attention_config
        pass

    def forward(self):
        pass

class DaoMLA(nn.Module):
    """
    MLA Attention
    """
    def __init__(self, attention_config):
        self.attention_config = attention_config
        pass

    def forward(self):
        pass

class DaoAttention(nn.Module):
    """
    MHA、GQA、MQA、MLA Attention
    """
    def __init__(self, attention_config):
        if attention_config.get("type") == "MHA":
            self.attention = DaoMHA(attention_config)
        elif attention_config.get("type") == "GQA":
            self.attention = DaoGQA(attention_config)
        elif attention_config.get("type") == "MQA":
            self.attention = DaoMQA(attention_config)
        elif attention_config.get("type") == "MLA":
            self.attention = DaoMLA(attention_config)
        else:
            raise ValueError(f"Invalid attention type: {attention_config.get('type')}")

    def forward(self, h, mask=None, pos_emb=None, past_key_value=None, use_cache=False):
        return self.attention(h, mask, pos_emb, past_key_value, use_cache)

class DaoFeedForward:
    def __init__(self):
        pass

    def forward(self):
        pass

class DaoMOE:
    def __init__(self):
        pass

    def forward(self):
        pass



class DaoBlock:
    def __init__(self):
        pass

    def forward(self):
        pass


class DaoLLM:
    def __init__(self):
        pass

    def forward(self):
        pass


class DaoCasualLLM:
    def __init__(self):
        pass

    def generate_response(self):
        """
        
        """
        pass