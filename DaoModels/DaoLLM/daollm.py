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
      
def rope_freq_cis(dim, max_len, theta=10000.0):
    """
    Args:
        dim: 每个token的维度
        max_len: 最大长度
        theta: 缩放因子
    Returns:
        cos: 余弦值
        sin: 正弦值
    """
    freqs = 1 / (theta ** ((torch.range(0, dim, 2)[:dim//2])/dim))      # 当维度为偶数时，不需要[:dim//2]；当维度是奇数的时候，不加切片freqs会多一项，当维度是奇数的时候，最后一个维度没有配对维度与之旋转，需要去掉。
    t = torch.arange(max_len, device=freqs.device)  # 每个token的位置
    freqs = torch.outer(t, freqs).float()  # 每个token的频率    （max_len, dim//2）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)     # (max_len, dim), 前一半与后一半相同
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)     # (max_len, dim), 前一半与后一半相同
    return freqs_cos, freqs_sin

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
        在rope的公式推导中，head_dim维度上两两分组，每组内进行rope操作。在实现中是前一半和后一半做rope操作，是因为rope_freq_cis中在计算相位的时候就是concat了前一半和后一半。所以第一个元素和第dim//2+1的元素位置的三角函数是配对的[[cos, -sin]，[sin, cos]]；只不过是之前计算的应该是q、k的相邻元素，现在变成了q、k的第1个元素与第dim//2+1个元素计算、第2个元素与第dim//2+2个元素计算（由于q、k在dim维度上所有维度元素是等价的，所以可以这样计算）。
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

        return self.o_proj(attn_output), past_key_value


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
    def __init__(self, dao_block_config):
        self.dao_block_config = dao_block_config

        attention_config = dao_block_config.get("attention")
        self.attention = DaoAttention(attention_config)

        feed_forward_config = dao_block_config.get("feed_forward")
        if feed_forward_config.get("forward_type") == "MOE":
            self.feed_forward = DaoMOE(feed_forward_config)
        elif feed_forward_config.get("forward_type") == "FFN":
            self.feed_forward = DaoFeedForward(feed_forward_config)
        else:
            raise ValueError(f"Invalid forward type: {feed_forward_config.get('forward_type')}")

        self.attn_norm = DaoLayerNorm(self.hidden_dim)
        self.ffn_norm = DaoLayerNorm(self.hidden_dim)

    def forward(self, h, mask=None, pos_emb=None, past_key_value=None, use_cache=False):
        residual = h
        attn_output, past_key_value = self.attention(self.attn_norm(h), mask, pos_emb, past_key_value, use_cache)
        h = residual + attn_output
        ffn_output = self.feed_forward(self.ffn_norm(h))
        h = residual + ffn_output
        return h, past_key_value


class DaoModel(nn.Module):
    def __init__(self, dao_model_config):
        self.dao_model_config = dao_model_config
        self.num_layers = dao_model_config.get("num_layers")
        self.vocab_size = dao_model_config.get("vocab_size")
        self.hidden_size = dao_model_config.get("hidden_size")
        self.max_length = dao_model_config.get("max_length")
        self.rope_theta = dao_model_config.get("rope_theta")

        self.dao_block_config = dao_model_config.get("dao_block")

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        freqs_cos, freqs_sin = rope_freq_cis(self.hidden_size, self.max_length)

        self.layers = nn.ModuleList([DaoBlock(self.dao_block_config) for _ in self.num_layers])

        self.head_layer_norm = DaoLayerNorm(self.hidden_size)
        self.head_layer = nn.Linear(self.hidden_size, self.vocab_size)

        # 将预计算的位置编码（freqs_cos和freqs_sin）注册为模型的缓冲区
        # 这样这些张量会被保存到模型状态中，但不会被视为模型参数（不会在反向传播中更新）
        # 使用register_buffer可以确保这些张量在模型移动到不同设备（如GPU）时自动跟随模型一起移动
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, input_ids, mask=None, past_key_value=None, use_cache=False):
        past_key_value = past_key_value or [None] * self.num_layers

        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        start_pos = past_key_value[0][0].shape[1] if past_key_value[0] is not None else 0

        h = self.embedding(input_ids)

        pos_emb = (
            self.freqs_cos[start_pos:start_pos+seq_len],
            self.freqs_sin[start_pos:start_pos+seq_len]
        )

        new_key_value = []
        for layer, past_key_value in zip(self.layers, past_key_value):
            h, past_key_value = layer(h, mask, pos_emb, past_key_value, use_cache)
            new_key_value.append(past_key_value)

        h = self.head_layer_norm(h)
        h = self.head_layer(h)

        return h, new_key_value

class DaoCasualLLM:
    def __init__(self):
        pass

    def generate_response(self):
        """
        
        """
        pass