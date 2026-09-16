$\pi_{0.5}$ ：
```
Background

However, the diversity of situations that a robot might encounter in the real world requires more than just scale.
How can we structure a training recipe for a robotic learning system that can enable this kind of flexible generalization?

Like human, generalization robotics might need to have the ability to transfer experience and knowledge from a variety of information sources. 

Oberservation:
by casting different modalities into the same sequence modeling framework, VLAs can be adapted to train on robot data, language data, computer vision tasks, and combinations of the above.

so solution: 
design a co-training framework
```
**co-training means?**
**通过多源数据混合训练，将知识迁移到目标移动操作平台上**。它不再仅依赖单一机器人的动作数据，而是整合了：
- **MM（移动操作数据）**：约400小时，在约100个家庭环境中收集（最接近评估场景）。
- **ME（多环境静态机械臂数据）**：在更多家庭中收集的非移动机械臂数据（跨本体）。
- **CE（跨本体实验室数据）**：包含多种机器人、多种桌面任务（含公开数据集OXE）。
- **HL（高层子任务预测数据）**：人工标注的语义子任务（如“捡起盘子”），训练模型进行链式推理。
- **WD（多模态网络数据）**：图像描述、VQA、目标检测等，提供广泛的物体语义知识。
- **VI（口头指令数据）**：后训练阶段加入，专家通过语言“遥控”机器人提供的高层示范。

Arc?
![[pi0.5_arc.png]]

分层推理
$$π_θ​(a_{t:t+H}​,\hat{l}∣o_t​,ℓ)=π_θ​(a_{t:t+H}​∣o_t​,\hat{l})⋅π_θ​(\hat{l}∣o_t​,ℓ)$$
先推理高层子任务，再根据子任务推理动作
Loss:
$$\mathcal{L} = \mathbb{E}_{\mathcal{D},\tau, w}[CE(x_{1:M}, f_\theta^t)+\alpha\|w -a_{t:t+H},- f_\theta^{a}(a^{\tau, w}_{t:t+H}, o_{t,}, l)\|]$$

**Training Strategy**:
>**预训练阶段（280k步）**：所有动作使用 **FAST离散令牌**（高效缩放），混合 MM、ME、CE、HL、WD 数据。

 >**后训练阶段（80k步）**：添加**动作专家（Action Expert）**，采用 **流匹配（Flow Matching）** 预测连续动作块（H=50步）。此时聚焦于 MM、ME、WD、HL 和 VI 数据，并过滤掉失败的短片段。

```
Results:
- **Q1：能否泛化到真实新家？**  
    在3个从未见过的真实家庭中进行厨房/卧室清理（持续2-5分钟），π0.5 成功完成任务（如将餐具放入水槽、衣物放入篮筐），其表现与在模拟测试环境中的结果高度一致。
    
- **Q2：泛化能力如何随场景数量扩展？**  
    训练环境数量从 3 个增加到 104 个时，四项任务（餐具入槽、物品入抽屉、衣物入篮、整理床铺）的平均性能持续提升。当使用104个环境时，性能接近直接包含测试环境训练的“作弊”基线。
    
- **Q3：各数据源（协同训练配方）有多重要？**（消融实验）
    
    - 移除 **ME 或 CE**（跨本体数据）会导致性能大幅下降。
        
    - 移除 **WD（网络数据）** 对常规任务影响不大，但会严重损害**分布外（OOD）物体**的语言跟随能力和高层推理能力。
        
    - 同时移除 ME 和 CE 性能最差。
        
- **Q4：对比其他 VLA（π0 和 π0-FAST+Flow）**  
    π0.5 显著优于二者。论文证实：**预训练使用离散FAST令牌 + 后训练使用流匹配**的混合范式，比纯扩散/流匹配训练更高效。
    
- **Q5：高层推理（HL）有多重要？**
    
    - 完整 π0.5（显式高层推理）表现最佳，甚至优于人类专家提供高层指令的“神谕”基线。
        
    - 有趣的是，**即使推理时不显式输出子任务（implicit HL）**，只要训练数据中包含 HL 数据，模型就能获得大部分收益，说明协同训练本身已注入推理能力。
        
    - 零样本使用 GPT-4 做高层规划效果最差，说明必须用机器人数据进行微调。
        
    - 移除 VI（口头指令）或 WD 会严重削弱高层策略。
```

$\pi_{0.6}$:
```
Background:
我们可以通过提示灵活地为通用机器人指定任务。但是就像人一样，这些模型需要练习技能才能掌握。这意味着不仅要利用演示数据，还要利用自主收集的经验数据，这些经验数据允许策略纠正其在部署中实际犯的错误，提高速度和鲁棒性，超越人类远程操作的水平，并适应新的部署条件。

Core:如何让 VLA 模型通过**真实世界的部署经验（试错）** 进行强化学习（RL）式自我提升，从而在长时程、灵巧操作任务（如叠衣服、做咖啡、折纸箱）中超越人类演示的性能上限？
```
introduce RL for finetuning

**Method**

- **策略模型 $\pi_{0.6}^*$**：负责输出具体的动作。它是一个基于 **Gemma 3 (4B)** 的 VLM 主干，外加一个 **860M 参数的动作专家（Action Expert）**。
- **价值模型 \(V_{\phi}\)（裁判）**：负责判断“当前状态距离成功还有多远”。它是一个较小的 VLM（**670M 参数**，也基于 Gemma 3），与策略网络结构相似但参数独立。

它们的交互关系如下（参考论文 Figure 3）：
策略网络在训练时，会**额外接收一个来自裁判的“优势指示器（Advantage Indicator \(I_t\)）”作为输入条件**，告诉模型当前这个动作是“好”还是“坏”。

Reward Function:
>**Defination**
	RECAP uses（Sparse Reward), 只在乎结局，不在乎过程细节：
$$
r_t = 
\begin{cases} 
0 & \text{如果 } t=T \text{ 且任务成功} \\
-C_{\text{fail}} & \text{如果 } t=T \text{ 且任务失败} \\
-1 & \text{其他情况（每多走一步扣1分）}
\end{cases}
$$
这个设计的目的是：**让价值函数学会预测“距离成功还有多少步”**（负值，越接近0越成功）。

Value Function
裁判要学习预测从当前状态到任务结束的累积回报 \(R_t(\tau)\)。论文使用了**分布式价值函数（Distributional Value Function）**，把回报值离散化成 201 个桶（bins），然后当作分类问题来训练：
$$
\min_{\phi} \mathbb{E}_{\tau \in \mathcal{D}} \left[ \sum_{\mathbf{o}_t \in \tau} H\left( R_t^B(\tau),\; p_\phi(V|\mathbf{o}_t, \ell) \right) \right]
$$

- $R_t^B(\tau)$：将真实累积回报离散化后的桶编号（标签）。
- $p_\phi(V|\mathbf{o}_t, \ell)$：裁判模型预测的落在每个桶上的概率分布。
- $H(\cdot, \cdot)$：交叉熵损失。

训练完成后，裁判会输出一个连续的期望价值 $V^{\pi_{\text{ref}}}(\mathbf{o}_t, \ell)$（将离散桶按概率加权求和）。

### 3. 核心公式二：运动员（策略网络）的架构与损失函数

**策略网络输入端的改造（优势指示器 \(I_t\)）：**
在文本输入中插入一个特殊的标记：“Advantage: positive” 或 “Advantage: negative”。

- \(I_t = 1\)（正）表示：这个动作的优势 \(A > \epsilon_\ell\)（比平均好）。
- \(I_t = 0\)（负）表示：这个动作比平均差。
- 对于**人工干预（专家修正）**的动作，直接强制设定 \(I_t = 1\)（专家一定是对的）。

**策略网络的输出端（三个输出）：**
1. **高层子任务 \(\hat{\ell}\)**（文本，自回归生成）。
2. **离散动作令牌 \(a^\ell_{t:t+H}\)**（FAST 分词器，辅助训练）。
3. **连续动作 \(\mathbf{a}_{t:t+H}\)**（流匹配 Action Expert 输出，用于实际控制）。

**最终的联合损失函数（公式 4 结合公式 3）：**
由于连续动作的流匹配无法直接计算对数似然，论文推导出它的证据下界（ELBO），最终整体的优化目标等价于：

\[
\mathcal{L} = \underbrace{-\log \pi_\theta(\hat{\ell}|\mathbf{o}_t, \ell)}_{\text{子任务交叉熵}} 
+ \underbrace{-\log \pi_\theta(a^\ell_{t:t+H}|\mathbf{o}_t, \ell, \hat{\ell})}_{\text{离散动作交叉熵 (FAST)}} 
+ \underbrace{\alpha \cdot \mathbb{E}_{\eta, \omega} \left[ \left\| \omega - \mathbf{a}_{1:H} - f_\theta(\mathbf{a}_{1:H}^{\eta, \omega}, I_t, \mathbf{o}_t, \ell, \hat{\ell}) \right\|^2 \right]}_{\text{流匹配 MSE（连续动作）}}
\]

**详细拆解这个流匹配项（最关键的部分）：**

- 输入给动作专家的带噪动作：\(\mathbf{a}_{1:H}^{\eta, \omega} = \eta \cdot \mathbf{a}_{1:H} + (1 - \eta) \cdot \omega\)
  - \(\mathbf{a}_{1:H}\)：真实动作序列（Ground Truth）。
  - \(\omega \sim \mathcal{N}(0, \mathbf{I})\)：标准高斯噪声。
  - \(\eta \in [0, 1]\)：流匹配的时间步（0 是纯噪声，1 是纯动作）。

- 模型要预测的目标（速度场）：\(\omega - \mathbf{a}_{1:H}\)（从噪声指向真实动作的矢量方向）。

- \(f_\theta(\cdots)\)：动作专家预测出的速度向量。

- **MSE 损失**：计算预测速度和理论真实速度之间的平方差。误差越小，模型越能精准地把噪声“雕刻”成真实动作。

---

### 4. 关键超参数：优势阈值 \(\epsilon_\ell\) 与优势计算

**优势值 \(A\) 的计算（见论文附录 F）：**
\[
A^{\pi}(\mathbf{o}_t, \mathbf{a}_t) = \sum_{t'=t}^{t+N-1} r_{t'} + V^{\pi}(\mathbf{o}_{t+N}) - V^{\pi}(\mathbf{o}_t)
\]
（\(N=50\) 步的 TD 估计，即当前动作带来的即时奖励加上未来状态的价值变化）

**阈值设定：**
- 在**预训练阶段**，设定阈值 \(\epsilon_\ell\)，使得数据集中约有 **30%** 的动作被标记为“正优势”（好的）。
- 在**后训练（微调）阶段**，根据任务调整阈值（比如让 10%~40% 的数据为正），控制模型向最优动作靠拢的激进程度。

---

### 5. 总结：架构和公式如何闭环？

1. **部署**策略 → 收集数据（成败标签）。
2. **裁判**（公式 1）学会判断当前状态的“价值”（剩余步数）。
3. 计算每个动作的**优势 \(A\)**（好动作 \(I=1\)，坏动作 \(I=0\)）。
4. **运动员**（公式 3+4）在训练时，看到 \(I=1\) 的动作会**加大模仿力度**，看到 \(I=0\) 的动作会**减小模仿力度**，从而策略在下一次部署时**更倾向于走那些被裁判认定为“优势为正”的轨迹**。

这就是 **RECAP（优势条件策略）** 用数学公式实现的灵魂：**不用复杂的策略梯度（PPO），只通过给数据打标签（正/负）并改变损失函数权重，就把强化学习的信号悄无声息地“喂”进了超大 VLA 模型中。** 这种设计极其轻量，适合训练千亿参数的流匹配模型，这也是它能工业化落地的核心原因。