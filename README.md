# FlashAttention

### （1）题目意思与生僻概念解释

这段代码实现的是一个概念版（非底层CUDA优化版）的 **FlashAttention**。

在传统的 Transformer 模型中，注意力机制（Self-Attention）需要计算 Query 矩阵和 Key 矩阵的点积，这会生成一个形状为 $N \times N$（$N$ 为序列长度）的巨大注意力分数矩阵。当序列极长时，这个矩阵会占用极为庞大的显存（内存复杂度为 $O(N^2)$），导致计算极慢甚至显存溢出（OOM）。

FlashAttention 的核心目标是**打破内存墙（Memory Wall）**。它利用硬件的内存层级特性，将庞大的运算拆分开来，从而大幅降低显存的读写开销。

**生僻概念解释：**

* **内存墙与 SRAM / HBM**：GPU 内部包含极小但极快的缓存（SRAM）以及庞大但较慢的显存（HBM）。标准注意力机制需要频繁地将庞大的 $N \times N$ 矩阵读写回较慢的 HBM。FlashAttention 的目的是尽量将数据保留在极快的 SRAM 中完成计算。
* **Tiling（分块/切片计算）**：一种硬件优化技术。由于完整的 $N \times N$ 矩阵无法装入极快的 SRAM 中，算法将原始的 Q、K、V 矩阵切割成固定大小的“块（Block）”，每次只将几个小块加载进 SRAM 进行运算。
* **Safe / Online Softmax（局部增量 Softmax）**：标准的 Softmax 函数需要获取一整行的所有数据才能找到最大值（用于防止数值溢出）并求总和（用于分母）。在分块计算中，每次只能看到部分数据。算法通过维护两个动态变量——当前的局部最大值（`max_scores`）和局部分母和（`softmax_sum`），实现了在分块计算时也能得出与全局计算完全一致的数学结果，极大地节省了内存。

---

### （2）代码解题思路解析

从代码结构进行逆推，该概念实现的底层逻辑如下：

1. **参数与线性映射**：首先定义 Q、K、V 的线性变换矩阵。这与标准注意力机制完全一致。
2. **空间重塑与多头切分**：将输入向量经过线性层后，切分为多头格式，以便进行并行特征提取。
3. **核心 Tiling（分块）双重循环结构**：
* **外层循环**：按照设定的 `block_size`（代码中为 64），依次截取 Key 和 Value 矩阵的块（`K_block`, `V_block`）。
* **局部变量初始化**：针对当前的块计算，初始化用于保存局部注意力得分、局部最大值、局部指数和的张量。
* **内层循环**：按照相同的 `block_size` 依次截取 Query 矩阵的块（`Q_block`）。
* **局部注意力与增量 Softmax 计算**：计算 `Q_block` 与 `K_block` 的内积。接着提取当前块的最大值，并更新全局的最大值张量 `max_scores`。基于动态更新的最大值，计算指数并累加到分母变量 `softmax_sum` 中。最终用指数得分除以当前的总和，得到局部归一化的权重。


4. **特征加权与整合**：将计算出的局部注意力权重与对应的 `V_block` 进行矩阵乘法，将结果累积到最终的输出张量 `out` 中。最后通过一个线性层进行输出融合。

*(注：提供的 Python 代码仅为展示 FlashAttention 局部 Softmax 计算思路的概念验证原型，其多重循环内的张量拼接与实际用于生产环境的底层 CUDA 实现存在逻辑结构上的差异，此处严格按照给定代码的执行流进行解析。)*

---

### （3）带有详细注释的代码

```python
import torch
import torch.nn.functional as F

class FlashAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, scale=None):
        super(FlashAttention, self).__init__()
        # 保存嵌入维度
        self.embed_dim = embed_dim
        # 保存注意力头数
        self.num_heads = num_heads
        # 计算单个注意力头的特征维度大小
        self.head_dim = embed_dim // num_heads
        # 确定缩放因子，若未提供则默认使用 head_dim 的负平方根
        self.scale = scale or self.head_dim ** -0.5
        
        # 初始化 Query, Key, Value 的线性映射网络层
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        # 初始化最终输出的线性映射网络层
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # 获取输入张量的批次大小、序列长度和嵌入维度
        batch_size, seq_len, embed_dim = x.size()
        
        # 计算 Q, K, V 矩阵
        # 经过线性层映射后，利用 view 拆分多头，并通过 transpose 将头维度置于前面
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 初始化一个与 Q 形状完全相同的全零张量，用于累积最终输出
        out = torch.zeros_like(Q)
        # 设定分块大小 (Tiling Block Size)，用于限制每次计算的显存占用
        block_size = 64  
        
        # 外层循环：遍历序列长度，每次跨度为 block_size，用于截取 K 和 V
        for i in range(0, seq_len, block_size):
            # 取当前序列块范围内的 Key 和 Value 子矩阵
            K_block = K[:, :, i:i + block_size]
            V_block = V[:, :, i:i + block_size]
            
            # 初始化当前块所需的注意力得分缓存矩阵
            attention_scores = torch.zeros(batch_size, self.num_heads, seq_len, block_size, device=x.device)
            # 初始化局部最大值缓存，初始设定为负无穷大
            max_scores = torch.full((batch_size, self.num_heads, seq_len, 1), float('-inf'), device=x.device)
            # 初始化 Softmax 计算所需的分母（指数和）缓存，初始设定为 0
            softmax_sum = torch.zeros(batch_size, self.num_heads, seq_len, 1, device=x.device)
            
            # 内层循环：遍历序列长度，截取 Query 矩阵的块
            for j in range(0, seq_len, block_size):
                # 取当前序列块范围内的 Query 子矩阵
                Q_block = Q[:, :, j:j + block_size]
                
                # 计算当前 Q_block 和 K_block 之间的相似度矩阵，并乘以缩放因子
                attention_chunk = torch.matmul(Q_block, K_block.transpose(-1, -2)) * self.scale
                
                # 使用局部 softmax 策略，提取当前 chunk 在最后一个维度上的最大值，并与历史最大值进行对比取大者
                max_scores = torch.maximum(max_scores, attention_chunk.max(dim=-1, keepdim=True)[0])
                # 计算当前 chunk 减去最大值后的自然指数，防止数值溢出
                exp_scores = torch.exp(attention_chunk - max_scores)
                # 将算得的指数值沿最后一个维度求和，并累加到分母缓存中
                softmax_sum += exp_scores.sum(dim=-1, keepdim=True)
                
                # 计算当前块的部分注意力权重（处于更新过程中的局部 softmax 值）
                attention_scores[:, :, j:j + block_size] = exp_scores / softmax_sum
                
            # 内层循环结束，利用计算完毕的注意力权重矩阵与 V_block 进行加权求和，累积到 out 对应的块中
            out[:, :, i:i + block_size] = torch.matmul(attention_scores, V_block)
            
        # 通过 transpose 恢复维度顺序，使用 contiguous 保证内存连续，最后重塑回合并多头的原始维度
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 通过最终的线性映射层整合特征并返回
        return self.out_proj(out)

# 示例输入与运行
# 构建随机输入，批次为 2，序列长度为 128，特征维度为 512
x = torch.randn(2, 128, 512)  
# 实例化 FlashAttention 模块，配置对应的维度参数
flash_attention = FlashAttention(embed_dim=512, num_heads=8)
# 执行前向计算并获取结果
output = flash_attention(x)

```

---

### （4）逐行详细中文解释

* **第1-2行**：导入 PyTorch 的核心模块及包含常用数学函数的模块。
* **第4行**：定义名为 `FlashAttention` 的类，继承自神经网络基础类 `nn.Module`。
* **第5行**：类的初始化函数，接收整体特征维度、注意力头数，以及可选的缩放因子参数。
* **第6行**：执行父类的初始化调用，使得网络参数能被自动管理。
* **第8-12行**：计算并保存模块的基础配置参数，其中单个头的特征维度等于总维度整除头数，缩放因子设定为单头维度大小的平方根的倒数（若未显式传入）。
* **第15-18行**：实例化四个全连接线性层，分别用于将输入数据投影到查询空间（Q）、键空间（K）、值空间（V）以及最终的输出空间。
* **第20行**：定义模块的前向传播执行逻辑。
* **第21行**：从输入张量中解构出批次大小、序列长度以及特征维度大小。
* **第24-26行**：对输入分别执行线性变换，运用 `view` 方法将最后的维度切分给多个注意力头，然后运用 `transpose(1, 2)` 将序列长度与头数的维度位置进行对调。处理后的张量形状变为 `[batch_size, num_heads, seq_len, head_dim]`。
* **第29行**：在设备内存中创建一个与 Q 形状、数据类型完全相同的填充为 0 的张量，作为最终加权特征聚合的容器。
* **第30行**：设定计算的切块大小为 64，意味着巨型矩阵将被切割成边长为 64 的小方块以控制内存占用。
* **第32行**：启动针对 Key 和 Value 的外层遍历，步长设定为切块大小。
* **第34-35行**：沿序列长度的维度（索引为2的维度），利用 Python 的切片语法裁剪出当前处于激活状态的 K 子块与 V 子块。
* **第38-42行**：在处理每个 K、V 块的初始阶段，开辟三个专用缓冲区。`attention_scores` 缓冲当前块的注意力分值矩阵；`max_scores` 存放每一行的局部最大值（初始化为负无穷）；`softmax_sum` 存放每一行的自然指数之和（初始化为零）。
* **第44行**：启动内层遍历，用于对 Query 矩阵进行分块提取。
* **第46行**：裁剪出当前处于激活状态的 Q 子块。
* **第49行**：执行高维矩阵的点积运算。调用转置操作对 K 子块的最后两个维度（即序列长度与特征维度）进行颠倒，随后与 Q 子块相乘并乘上缩放标量，得到未归一化的原始分块分数 `attention_chunk`。
* **第52行**：实施增量 Softmax 的关键第一步。调用 `max(dim=-1)` 取出新算出的块在横向上的最大值，接着运用 `torch.maximum` 与历史最高记录进行比较，刷新并记录当前所见过的绝对最大值。
* **第54行**：实施增量 Softmax 的第二步。将当前块的所有分数统一减去刚算得的安全最大值，再进行 `torch.exp` 指数化，保障浮点运算安全稳定。
* **第55行**：将新算得的指数值在横向上进行累加，并将结果补充至总体求和容器 `softmax_sum` 内。
* **第58行**：将当前 Q 子块对应的指数矩阵除以此刻的总和，得出局部的 Softmax 权重，写入预先分配的 `attention_scores` 切片槽位中。
* **第61行**：内层所有 Q 块遍历结束，当前 K、V 块的权重已计算就绪。此时利用算出的权重矩阵与对应的 `V_block` 进行加权内积，并将产出的新特征块赋值给 `out` 矩阵对应的分块区域。
* **第64行**：所有块运算终结。对最终合成的 `out` 张量实施维度回退操作：置换首尾位置、确保数据物理地址连续排列，最后揉合多头维度还原至单一特征向量的形态。
* **第65行**：把还原后的特征穿过末端全连接层，随即弹射回调用方。
* **第68-72行**：给出实例用法的构建脚本，模拟数据流生成及模块的装载启动过程。

---

### （5）具体数值算例与追踪过程表格

设定一个极小环境用于模拟单次循环执行的数值流转（使内外层循环均只执行 1 次）：

* `batch_size = 1`
* `num_heads = 1`
* `seq_len = 2`
* `embed_dim = 2` (因此 `head_dim = 2`，`scale = 0.707`)
* `block_size = 2`（该设定下切块等于全量处理）

假设映射后的 Q、K、V 均为 `[[[ [1.0, 0.0], [0.0, 1.0] ]]]`。形状为 `(1, 1, 2, 2)`。

| 步骤 | 变量/张量形状及数值变化 | 对应代码行 | 对应代码 |
| --- | --- | --- | --- |
| **初始准备** | `Q`, `K`, `V` 形状皆为 `(1, 1, 2, 2)`<br>

<br>值皆为: `[[ [[1,0], [0,1]] ]]` | 行 24-26 | `Q = ...`, `K = ...`, `V = ...` |
| **外层切片** | `i = 0`。提取完整的矩阵为块。<br>

<br>`K_block` = `[[ [[1,0], [0,1]] ]]`<br>

<br>`V_block` = `[[ [[1,0], [0,1]] ]]` | 行 34-35 | `K_block = K[:, :, i:i + block_size]`<br>

<br>`V_block = ...` |
| **缓冲初始化** | `max_scores` = `[[[[-inf], [-inf]]]]`<br>

<br>`softmax_sum` = `[[[[0], [0]]]]` | 行 41-42 | `max_scores = torch.full(...)`<br>

<br>`softmax_sum = torch.zeros(...)` |
| **内层切片** | `j = 0`。提取块:<br>

<br>`Q_block` = `[[ [[1,0], [0,1]] ]]` | 行 46 | `Q_block = Q[:, :, j:j + block_size]` |
| **计算内积** | 计算点积并乘以 $0.707$。<br>

<br> `[[1,0], [0,1]]` $\times$ `[[1,0], [0,1]]`<br>

<br> $\rightarrow$ `attention_chunk`: `[[ [[0.707, 0], [0, 0.707]] ]]` | 行 49 | `attention_chunk = torch.matmul(...) * self.scale` |
| **增量更新 Max** | 各行的最大值为 $0.707$。<br>

<br> $\rightarrow$ `max_scores`: `[[ [[0.707], [0.707]] ]]` | 行 52 | `max_scores = torch.maximum(...)` |
| **安全指数化** | 当前项减去 max 后进行 exp。<br>

<br> 行0: $e^{0.707-0.707}=1$, $e^{0-0.707}\approx0.493$<br>

<br> 行1: $e^{0-0.707}\approx0.493$, $e^{0.707-0.707}=1$<br>

<br> $\rightarrow$ `exp_scores`: `[[ [[1, 0.493], [0.493, 1]] ]]` | 行 54 | `exp_scores = torch.exp(...)` |
| **增量求分母** | 各行求和累加。<br>

<br> 行0: $1 + 0.493 = 1.493$<br>

<br> 行1: $0.493 + 1 = 1.493$<br>

<br> $\rightarrow$ `softmax_sum`: `[[ [[1.493], [1.493]] ]]` | 行 55 | `softmax_sum += exp_scores.sum(...)` |
| **计算局部权重** | 除以总和得权重。<br>

<br> 行0: $1/1.493 \approx 0.67$, $0.493/1.493 \approx 0.33$<br>

<br> $\rightarrow$ `attention_scores`: `[[ [[0.67, 0.33], [0.33, 0.67]] ]]` | 行 58 | `attention_scores[...] = exp_scores / softmax_sum` |
| **特征乘加输出** | 权重矩阵乘以 `V_block`。<br>

<br> $\rightarrow$ `out`: `[[ [[0.67, 0.33], [0.33, 0.67]] ]]`<br>

<br>随后还原维度并输送至线性层。 | 行 61 | `out[:, :, i:i + block_size] = torch.matmul(...)` |
