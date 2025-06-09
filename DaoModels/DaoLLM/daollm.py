import torch
import torch.nn as nn
import math



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
      
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embedding to the query and key tensors.
    Args:
        q: query tensor of shape (batch_size, head_num, seq_len, head_dim)
        k: key tensor of shape (batch_size, head_num, seq_len, head_dim)
        cos: cosine tensor of shape (seq_len, head_dim)
        sin: sine tensor of shape (seq_len, head_dim)
    Returns:
        q_emb: query tensor of shape (batch_size, head_num, seq_len, head_dim)
        k_emb: key tensor of shape (batch_size, head_num, seq_len, head_dim)
    Returns:

    explain:
        rope作用的维度是head_dim维度上的所有元素。
        在rope的公式推导中，head_dim维度上两两分组，每组内进行rope操作。在实现中是前一半和后一半做rope操作，是因为head_dim维度上的所有元素相互独立，是等价的，所以任意两个元素都可以做rope操作，不需要相邻元素。
    """
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos, sin = cos.to(q.device), sin.to(q.device)
    q_emb = q * cos + rotate_half(q) * sin
    k_emb = k * cos + rotate_half(k) * sin
    return q_emb, k_emb

class DaoMHA(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, attention_config):
        self.attention_config = attention_config
        self.head_dim = attention_config.get("head_dim")
        self.num_heads = attention_config.get("num_heads")
        self.q_proj = nn.Linear(attention_config.get("hidden_dim"), self.head_dim * self.num_heads)
        self.k_proj = nn.Linear(attention_config.get("hidden_dim"), self.head_dim * self.num_heads)
        self.v_proj = nn.Linear(attention_config.get("hidden_dim"), self.head_dim * self.num_heads)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, attention_config.get("hidden_dim"))

    def forward(self, h, mask=None, pos_emb=None, past_key_value=None, use_cache=False):
        """
        Args:
            h: batch * seq_len * hidden_dim
            mask: batch * seq_len
            pos_emb: batch * seq_len * head_dim
        """
        batch_size, seq_len, hidden_dim = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attention_config.get("pos_emb_type") == "rope":
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self.attention_config.get("pos_emb_type") == "alibi":
            pass
        elif self.attention_config.get("pos_emb_type") == "learnable":
            q += pos_emb
            k += pos_emb
        else:
            raise ValueError(f"Invalid pos_emb_type: {self.attention_config.get('pos_emb_type')}")

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        if use_cache:
            past_key_value = (k, v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = torch.softmax(attn_weights, dim=-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_output = torch.matmul(attn_scores, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


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
    """
    GLU和FFN的区别：
    1. 网络结构变化：
    原始GLU结构：使用了三个线性层（gate_proj、up_proj、down_proj），其中门控机制通过 gate_proj(x) * up_proj(x) 实现
    传统FFN结构：只使用两个线性层（linear1、linear2），结构更简单直接
    2. 中间维度计算：
    原始：int(config.hidden_size * 8 / 3) ≈ 2.67倍隐藏维度
    传统FFN：config.hidden_size * 4 = 4倍隐藏维度（这是Transformer中更常见的设置）
    3. 前向传播流程：
    原始GLU：dropout(down_proj(act_fn(gate_proj(x)) * up_proj(x)))
    传统FFN：dropout(linear2(dropout(act_fn(linear1(x)))))
    4. 计算效率：
    GLU：需要计算两个并行的线性变换然后相乘，参数量更多
    传统FFN：顺序计算，结构更简单，参数量相对较少
    """
    def __init__(self, feed_forward_config):
        self.feed_forward_config = feed_forward_config
        self.hidden_dim = feed_forward_config.get("hidden_dim")
        self.intermediate_dim = feed_forward_config.get("intermediate_dim")
        self.dropout = feed_forward_config.get("dropout")
        self.activation = feed_forward_config.get("activation")
        self.up = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.down = nn.Linear(self.intermediate_dim, self.hidden_dim)

        if self.feed_forward_config.get("forward_type") == "GLU":
            self.gate = nn.Linear(self.hidden_dim, self.intermediate_dim)
        elif self.feed_forward_config.get("forward_type") == "FFN":
            pass
        else:
            raise ValueError(f"Invalid forward type: {self.feed_forward_config.get('forward_type')}")

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, h):
        if self.feed_forward_config.get("forward_type") == "GLU":
            h = self.up(h) * self.activation(self.gate(h))
            h = self.down(h)
        elif self.feed_forward_config.get("forward_type") == "FFN":
            h = self.down(self.dropout(self.activation(self.up(h))))
        else:
            raise ValueError(f"Invalid forward type: {self.feed_forward_config.get('forward_type')}")
        return h

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