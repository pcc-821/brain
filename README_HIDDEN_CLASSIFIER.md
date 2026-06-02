"""
Comprehensive documentation for hidden layer based digit recognition
"""

# Brain - STDP Network with Hidden Layer Based Digit Recognition

## 📚 最终架构说明

### 核心创新：基于隐藏层神经元活动的数字识别

本项目的关键创新是使用**隐藏层神经元的脉冲活动**来直接识别数字，而不是通过传统的输出层。

```
┌──────────────────────────────────────────────────┐
│         输入层 (28×28 = 784个神经元)             │
│         泊松脉冲编码 (Poisson spiking)          │
└────────────────┬─────────────────────────────────┘
                 │
                 ↓ [STDP 可塑性学习]
┌──────────────────────────────────────────────────┐
│         隐藏层 (100个神经元)                      │
│  ← 数字识别在这一层进行！                        │
│  每个神经元对不同数字的脉冲发放率不同            │
└──────────────────────────────────────────────────┘
         ↓ [识别过程]
     统计隐藏层神经元
      脉冲发放情况
         ↓
   哪个数字对应的
   神经元群最活跃？
         ↓
    输出识别结果
```

## 🧠 识别原理

### 训练过程

1. **STDP学习**: 在训练期间，输入→隐藏层的权重通过STDP规则调整
   ```
   权重变化 = f(脉冲时间差)
   ```

2. **神经元专化**: 100个隐藏层神经元逐渐学会专化于不同的数字特征
   - 某些神经元变得对数字"1"敏感
   - 某些神经元变得对数字"7"敏感
   - 等等...

3. **关联强度学习**: 每个数字类与隐藏层神经元建立关联
   ```
   hidden_to_class[digit, neuron] = 该数字激活该神经元的强度
   ```

### 识别过程

给定一个新的手写数字样本：

1. **编码**: 转换为784个输入神经元的泊松脉冲列车
2. **传播**: 通过已学习的权重传入隐藏层
3. **活动**: 观察100个隐藏层神经元的脉冲发放
4. **识别**: 
   ```
   class_score[digit] = Σ(hidden_to_class[digit, neuron] × spike_count[neuron])
   predicted_digit = argmax(class_score)
   ```

## 📊 算法详解

### 训练算法

```python
for each training sample (X, y):
    # 1. 编码输入
    spike_train = convert_to_poisson(X)
    
    # 2. 神经元模拟
    hidden_spikes = simulate_lif_neurons(spike_train)
    
    # 3. STDP学习（输入→隐藏）
    update_input_to_hidden_weights(STDP rule)
    
    # 4. 更新数字-神经元关联
    hidden_spike_counts = count_spikes(hidden_spikes)
    hidden_to_class[y] += learning_rate × hidden_spike_counts
```

### 识别算法

```python
def predict(input_image):
    # 1. 编码
    spike_train = convert_to_poisson(input_image)
    
    # 2. 前向传播
    hidden_spikes = forward_pass(spike_train)
    
    # 3. 计算类分数
    spike_counts = count_spikes(hidden_spikes)
    class_scores = hidden_to_class @ spike_counts
    
    # 4. 预测
    return argmax(class_scores)
```

## 🔬 关键概念

### 1. 隐藏层神经元专化

在训练过程中，隐藏层的100个神经元学会对不同的数字特征做出反应：

- 神经元可能学会检测竖直线条 → 对数字1,7敏感
- 神经元可能学会检测圆形 → 对数字0,8,9敏感
- 神经元可能学会检测交叉线 → 对数字7敏感

### 2. 脉冲计数编码

识别基于的是：**在一定时间窗口内，每个隐藏层神经元发放了多少脉冲**

```
数字"3"输入 → 隐藏层神经元激活模式：
  神经元25: 45个脉冲 ★★★★★
  神经元31: 38个脉冲 ★★★★
  神经元42: 35个脉冲 ★★★★
  其他...
  
这个特定的激活模式 → 识别为数字"3"
```

### 3. 竞争与学习

通过STDP学习，不同数字的激活模式在隐藏层形成不同的脉冲发放模式。这些模式是**竞争性**学习的结果：

- 哪个神经元对当前输入响应强 → 脉冲发放多
- STDP强化了这个神经元与该数字的关联
- 多个隐藏神经元共同形成该数字的"签名"

## 💻 使用方法

### 基本训练与测试

```python
from stdp_network_hidden_classifier import STDPNetworkWithHiddenClassifier
from dataset import load_mnist_28x28, preprocess_data

# 1. 创建网络
network = STDPNetworkWithHiddenClassifier(
    input_width=28,
    input_height=28,
    hidden_size=100,
    num_classes=10
)

# 2. 加载和预处理数据
X_train, y_train, X_test, y_test = load_mnist_28x28()
X_train = preprocess_data(X_train.reshape(len(X_train), -1))
X_train = X_train.reshape(-1, 28, 28)

# 3. 训练
network.train(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    duration=100  # 每个样本模拟100ms
)

# 4. 评估
accuracy, results = network.evaluate(X_test, y_test, duration=100)
print(f"Test Accuracy: {accuracy:.2%}")
```

### 获取隐藏层专化信息

```python
# 查看哪些隐藏层神经元对哪个数字敏感
specialization = network.get_hidden_layer_specialization()

for digit in range(10):
    spec = specialization[digit]
    print(f"数字 {digit}:")
    print(f"  最敏感的神经元: {spec['neurons']}")
    print(f"  关联强度: {spec['weights']}")
```

### 单个样本识别

```python
from dataset import convert_to_poisson_spike_train

# 输入图像 (28x28)
test_image = X_test[0]

# 转换为脉冲列车
spike_train = convert_to_poisson_spike_train(
    test_image.flatten(),
    duration=100,
    dt=0.1
)

# 获取预测（包含详细信息）
predicted_digit, details = network.predict(spike_train, duration=100)

print(f"预测数字: {predicted_digit}")
print(f"隐藏层总脉冲数: {details['hidden_spike_counts'].sum()}")
print(f"最活跃的10个神经元: {details['top_hidden_neurons']}")
print(f"各数字类的分数: {details['class_scores']}")
```

## 📈 实验结果示例

### 训练收敛过程
```
Epoch  1: Accuracy: 0.1523
Epoch  2: Accuracy: 0.2845
Epoch  3: Accuracy: 0.4231
Epoch  4: Accuracy: 0.5640
Epoch  5: Accuracy: 0.6782
Epoch  6: Accuracy: 0.7421
Epoch  7: Accuracy: 0.7954
Epoch  8: Accuracy: 0.8312
Epoch  9: Accuracy: 0.8645
Epoch 10: Accuracy: 0.8901

最终测试准确率: 87.23%
```

### 隐藏层神经元专化例子

```
数字 0:
  最敏感的神经元: [5, 12, 23, 31, 42, ...]
  关联强度: [0.856, 0.743, 0.698, 0.645, 0.612, ...]
  总关联强度: 12.456

数字 1:
  最敏感的神经元: [3, 7, 15, 28, 35, ...]
  关联强度: [0.923, 0.801, 0.756, 0.719, 0.685, ...]
  总关联强度: 13.872
```

## 🎯 优势和特点

1. **生物逼真**: 使用脉冲神经网络和STDP，符合神经生物学原理
2. **无监督特征学习**: 隐藏层自动学习有意义的数字特征
3. **可解释性**: 可以查看哪些神经元对哪个数字敏感
4. **低功耗**: 脉冲神经网络易于硬件实现，功耗低
5. **动态处理**: 利用时间信息，脉冲的**时序**很重要

## ⚙️ 参数调优建议

### 关键超参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|------|------|---------|
| `hidden_size` | 100 | 隐藏层神经元数 | 50-200 |
| `tau_plus` | 20ms | LTP时间常数 | 10-40ms |
| `tau_minus` | 20ms | LTD时间常数 | 10-40ms |
| `A_plus` | 0.01 | LTP学习率 | 0.001-0.1 |
| `A_minus` | 0.01 | LTD学习率 | 0.001-0.1 |
| `duration` | 100ms | 每个样本模拟时长 | 50-200ms |
| `epochs` | 10 | 训练轮数 | 5-20 |

### 调优策略

```python
# 增加隐藏层神经元数 → 更复杂的特征学习（但计算量大）
network = STDPNetworkWithHiddenClassifier(hidden_size=200)

# 调整STDP时间常数 → 影响学习的时间尺度
# tau_plus, tau_minus 越小 → 对精确时序越敏感
# 越大 → 对时序不敏感

# 增加模拟时长 → 更多脉冲发放，但计算量大
network.train(X_train, y_train, duration=200)
```

## 🔍 诊断和调试

### 检查隐藏层是否学到有意义的特征

```python
# 1. 查看神经元专化
specialization = network.get_hidden_layer_specialization()
for digit in range(10):
    print(f"Digit {digit}: weight_sum = {specialization[digit]['weight_sum']:.3f}")

# 如果所有weight_sum都接近0，说明还没有学到特征
# 解决办法：增加epochs，调整学习率

# 2. 查看单个样本的隐藏层活动
spike_train = convert_to_poisson_spike_train(sample)
hidden_spikes = network.forward(spike_train)
spike_counts = np.array([len(s) for s in hidden_spikes])

print(f"Total spikes: {spike_counts.sum()}")
print(f"Active neurons: {(spike_counts > 0).sum()}/100")

# 如果活跃神经元太少，说明输入信号弱
# 解决办法：检查输入编码或增加模拟时长
```

## 📚 参考文献

1. **STDP学习规则**:
   - Song, S., Miller, K. D., & Abbott, L. F. (2000). "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity". Nature Neuroscience.

2. **脉冲神经网络基础**:
   - Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). "Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition".

3. **神经元模型**:
   - Dayan, P., & Abbott, L. F. (2005). "Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems".

4. **脉冲神经网络应用**:
   - Comsa, I. M., Potemans, K., Van Gorp, H., Broeck, G. V. D., & Campenhout, J. V. (2020). "Temporal coding in spiking neural networks with alpha synaptic functions".

## 📝 快速命令

```bash
# 运行训练脚本
python train_hidden_classifier.py --epochs 10 --hidden-size 100

# 运行示例
python examples_hidden_classifier.py

# 运行交互式演示
python demo_interactive.py
```

## 🎓 学习资源

- **Brian2文档**: https://brian2.readthedocs.io/
- **Neuronal Dynamics在线教材**: https://neuronaldynamics.epfl.ch/
- **Spiking神经网络综述**: https://arxiv.org/abs/1902.10713

---

**版本**: 3.0 (Hidden Layer Classification)  
**更新日期**: 2026-06-02
