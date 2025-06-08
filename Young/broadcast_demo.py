import torch
import numpy as np

def print_tensor_info(tensor, name):
    """打印张量的详细信息"""
    print(f"{name}:")
    print(f"  形状: {tensor.shape}")
    print(f"  数据: {tensor}")
    print(f"  维度数: {tensor.ndim}")
    print()

def demonstrate_broadcast_rules():
    """演示PyTorch broadcast的基本规则"""
    print("=" * 60)
    print("PyTorch Broadcast 规则演示")
    print("=" * 60)
    
    print("规则1: 从右边开始对齐维度")
    print("-" * 30)
    
    # 示例1: 标量与张量
    a = torch.tensor(5.0)  # 标量
    b = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3张量
    
    print_tensor_info(a, "a (标量)")
    print_tensor_info(b, "b (2x3张量)")
    
    result = a * b  # 标量会broadcast到所有元素
    print_tensor_info(result, "a * b 的结果")
    
    print("规则2: 维度为1的会被扩展")
    print("-" * 30)
    
    # 示例2: 不同维度的张量
    c = torch.tensor([[1], [2]])  # 2x1张量
    d = torch.tensor([10, 20, 30])  # 1x3张量
    
    print_tensor_info(c, "c (2x1张量)")
    print_tensor_info(d, "d (1x3张量)")
    
    result2 = c + d  # c的第二维扩展为3，d的第一维扩展为2
    print_tensor_info(result2, "c + d 的结果")
    
    print("规则3: 缺失的维度会在左边补1")
    print("-" * 30)
    
    # 示例3: 维度数不同的张量
    e = torch.tensor([1, 2, 3])  # (3,)
    f = torch.tensor([[[10]], [[20]]])  # (2, 1, 1)
    
    print_tensor_info(e, "e (3,)")
    print_tensor_info(f, "f (2, 1, 1)")
    
    result3 = e * f  # e会变成(1, 1, 3)，然后broadcast
    print_tensor_info(result3, "e * f 的结果")

def demonstrate_layer_norm_broadcast():
    """演示LayerNorm中的broadcast"""
    print("=" * 60)
    print("LayerNorm中的Broadcast演示")
    print("=" * 60)
    
    # 模拟你的DaoLayerNorm中的情况
    batch_size = 2  # 批次大小
    seq_len = 4     # 序列长度
    dim = 3         # 特征维度
    
    # 创建输入张量h
    h = torch.randn(batch_size, seq_len, dim)
    print_tensor_info(h, "输入h (batch_size, seq_len, dim)")
    
    # 创建gamma参数（就像你代码中的self.gamma）
    gamma = torch.ones(dim)  # 形状为(dim,)
    print_tensor_info(gamma, "gamma参数 (dim,)")
    
    # 计算RMS
    h_squared = h.pow(2)
    print_tensor_info(h_squared, "h.pow(2)")
    
    mean_squared = h_squared.mean(-1, keepdim=True)
    print_tensor_info(mean_squared, "h.pow(2).mean(-1, keepdim=True)")
    
    rsqrt_mean = torch.rsqrt(mean_squared + 1e-6)
    print_tensor_info(rsqrt_mean, "torch.rsqrt(mean_squared + eps)")
    
    # 归一化
    normalized_h = h * rsqrt_mean
    print_tensor_info(normalized_h, "h * rsqrt_mean")
    
    # 应用gamma（这里发生broadcast）
    print("关键的broadcast步骤:")
    print(f"gamma形状: {gamma.shape}")
    print(f"normalized_h形状: {normalized_h.shape}")
    print("broadcast过程:")
    print("  gamma: (3,) -> (1, 1, 3) -> (2, 4, 3)")
    print("  normalized_h: (2, 4, 3)")
    
    final_result = gamma * normalized_h
    print_tensor_info(final_result, "最终结果: gamma * normalized_h")

def demonstrate_complex_broadcast():
    """演示复杂的broadcast场景"""
    print("=" * 60)
    print("复杂Broadcast场景演示")
    print("=" * 60)
    
    # 场景1: 多维broadcast
    print("场景1: 多维broadcast")
    print("-" * 30)
    
    a = torch.randn(8, 1, 6, 1)  # (8, 1, 6, 1)
    b = torch.randn(7, 1, 5)     # (7, 1, 5)
    
    print_tensor_info(a, "a")
    print_tensor_info(b, "b")
    
    # broadcast规则应用:
    # a: (8, 1, 6, 1) -> (8, 7, 6, 5)
    # b: (7, 1, 5) -> (1, 7, 1, 5) -> (8, 7, 6, 5)
    result = a + b
    print_tensor_info(result, "a + b 的结果")
    
    # 场景2: 注意力机制中的broadcast
    print("场景2: 注意力机制中的broadcast")
    print("-" * 30)
    
    # 模拟注意力分数
    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    
    # Query, Key, Value
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print_tensor_info(Q, "Query")
    print_tensor_info(K, "Key")
    
    # 计算注意力分数 Q @ K^T
    attention_scores = Q @ K.transpose(-2, -1)  # (batch, heads, seq_len, seq_len)
    print_tensor_info(attention_scores, "注意力分数")
    
    # 缩放因子（标量broadcast）
    scale = 1.0 / (head_dim ** 0.5)
    scaled_scores = attention_scores * scale
    print(f"缩放因子: {scale} (标量)")
    print_tensor_info(scaled_scores, "缩放后的注意力分数")

def demonstrate_broadcast_errors():
    """演示broadcast失败的情况"""
    print("=" * 60)
    print("Broadcast失败的情况")
    print("=" * 60)
    
    print("以下操作会失败，因为维度不兼容:")
    
    try:
        a = torch.randn(3, 4)  # (3, 4)
        b = torch.randn(2, 3)  # (2, 3)
        result = a + b  # 这会失败
    except RuntimeError as e:
        print(f"错误: {e}")
        print("原因: (3, 4) 和 (2, 3) 无法broadcast")
        print("  从右边对齐: 4 vs 3 (不兼容)")
        print("  从左边对齐: 3 vs 2 (不兼容)")
    
    print("\n正确的做法:")
    a = torch.randn(3, 4)  # (3, 4)
    b = torch.randn(1, 4)  # (1, 4) - 第一维为1，可以broadcast
    result = a + b
    print_tensor_info(a, "a (3, 4)")
    print_tensor_info(b, "b (1, 4)")
    print_tensor_info(result, "a + b 的结果")

def main():
    """主函数，运行所有演示"""
    print("PyTorch Broadcast 完整演示")
    print("作者: Young")
    print("=" * 60)
    
    # 运行所有演示
    demonstrate_broadcast_rules()
    demonstrate_layer_norm_broadcast()
    demonstrate_complex_broadcast()
    demonstrate_broadcast_errors()
    
    print("=" * 60)
    print("总结:")
    print("1. Broadcast从右边开始对齐维度")
    print("2. 维度为1的会被扩展到匹配的大小")
    print("3. 缺失的维度会在左边补1")
    print("4. 不兼容的维度会导致错误")
    print("5. 标量可以与任何形状的张量broadcast")
    print("=" * 60)

if __name__ == "__main__":
    main()