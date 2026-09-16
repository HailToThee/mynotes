From gradient policy to PPO: https://zhuanlan.zhihu.com/p/614115887
Flow matching: https://zhuanlan.zhihu.com/p/4116861550
DDPM: https://zhuanlan.zhihu.com/p/563661713
DiT: https://zhuanlan.zhihu.com/p/711055614  https://github.com/facebookresearch/DiT

### Action data transformation:
收集到的数据大多为图像和视频数据， 无法直接对action进行训练，因为flow matching所学习的$$z_{t} = (1-s)\epsilon+s z_{t+1}$$
$$v_{\theta}(z, s|z_{\leq t}, a_{\leq t}) = z_{t+1} - \epsilon$$
$$\min_{\theta}\|v_{\theta}- (z_{t+1}-\epsilon)\|^2$$
在latent生成路径上， 如何从随机噪声走到 $z_{t+1}$, 即标准速度场
并非 $z_{t+1} - z_{t}$, 也不是机器人的动作，因此通过直接学习插值的办法是无法获取正确动作的
所以 需要预先得到动作的chunk， 对这个chunk进行训练，minimize MSE loss.

如何获取动作chunk $a_t$
1.  **逆动力学模型(IDM)**: Train an extra Inverse Dynamic model $\phi$，input:  $z_{t ,}z_{t+1}$   
	output: $$l_{t} \quad where \quad d_\ell\ll \dim(z_t)$$check if the output $l_t$ is good enough using another model   $\psi$   $\hat z_{t+1}=f_\psi(z_t,\ell_t)$
2.  **逆运动学**: 转换为MANO 参数和手部关键点，获取对应手指的三维位置， 参考Qwen-RobotManip:https://www.alphaxiv.org/pdf/2606.17846v2
3. **机器人数据**：直接使用公开机器人数据集中的关节状态、末端执行器位姿和夹爪状态作为动作标签。
### VLA architecture
视觉-语言-动作模型（VLA）沿一条架构轴线分化：VLM是直接生成动作(自回归AR)，还是仅作为条件输入来调控一个独立的动作模块(VLM as Encoder)

连续动作头适合平滑的高频控制，自回归则更擅长推理且实现更简单

 1. **主流路线**是将预训练的VLM与一个动作专家（Action Expert）耦合。动作专家是一个专门负责生成动作的小型网络，它接收VLM提取的图像和语言特征，然后通过扩散或流匹配输出连续动作（连续动作是指像关节角度、末端位置这样的数值，可以平滑变化）。这条路线的代表模型有π₀、π₀.5、GR00T-N1等。


![[VLA _arc.png]]
	1. Vision Encoder 
		SigLIP or DINO， 多视角输入。
	2. Language-Action Backbone
		Action Expert, 前面VLM所处理得到的信息
			- DiT(VLM hidden state as KV cache, state + noisy actions + timestep(may concat))
			like QWEN, 
			- MoE Enhanced Transformer: hidden state
	3. Action Decoder
			- Diffusion Policy
			- Flow matching
	Anti-forgetting:
		Action learning knowaledge overritFes the VLM knowledge.
		Solution: Knowledge Insulation: Block the back propagation from Action Expert to VLM, so that VLM knowledge is not overwritten by Action Expert learning.

2. **自回归路线**则是将连续动作离散化（比如把"位移5cm"变成一个编号），然后让VLM像预测下一个文字一样，用"下一词元预测"的方式逐个生成动作编号。这条路线包括RT-2、OpenVLA等模型。
	Question 1: Action tokneization
		1. Per-timestep binning:
		2. DCT: 
		3. Cross-embodiment generalization: 同前两者不同，映射部位到指定的槽位(多个dim)里面实现控制，即映射动作而不是映射空间
	
	Question 2: Reasoning-action Alignment
		1. *Bolt-on CoT*: 
			image+prompt --> VLM --> plan&subtassks&2d route --> Controller --> Action
			VLM output必须被Controller正确解释，推理中的目标和位置必须和动作空间对应，高层并不直接生成动作。
		2. *In-stream CoT*
			image+prompt+state --> AR decoder --> Subtask / BBox / Trace / ActionHint --> [==ActionCodec==](ActionCodec)(from spatial token to action) --> Action
			推理token和action token在同一个autoregressive decoder中生成，位于同一条token sequence中，使用同一个词表和loss训练让推理和动作在**表示、目标函数以及生成时序上共同对齐**。

3. **WAM**: 要首先image the future, 然后才能决定如何行动。
	1. BackBone: 不依赖VLM作为主干，Use Video DiT to predict future video frames, then(or Jointly) predict actions based on predicted future frames.
	2. Input: 
		- observation(or plus prior frames)
		- language prompt.
		- random noises
	3. output:
	    - future video latent embedding 
        - actions
    4. Action Decode:  Flow Matching or MSE.
	5. 内部的三大路径(究竟是否要生成视频帧？)
#### Training Stages
1. Qwen-RoboManip:
	1. **数据对齐与统一表示**：80维规范向量 + 相机帧Delta位姿 + 五阶段筛选。
	2. **双流协同预训练**：VLA流（动作）+ VLM流（感知），9:1比例，损失 = L_FM + 0.1*L_VLM。
	3. **领域SFT与部署**：目标域微调，可选混合预训练数据防过拟合，RTC异步推理。
2. LingBot-VA2.0:
	1. Tokenizer training: 单独训练语义视觉-动作分词器，产出包含潜在动作的紧凑潜变量空间。Loss: IDM  FDM
	2. Video-action policy training: 在主DiT中训练因果视频-动作策略，将Tokenizer输出的潜变量作为监督信号，并引入MCP（多块预测）、ICL、HCT等高级策略。
	3. Distillation alignment: 将训练好的Policy通过蒸馏转化为实时控制器
#### In-context Learning(ICL)

**Demonstration-based ICL**: 在推理时，向模型输入**几个成功完成任务的“视觉-语言-动作”演示片段**作为上下文。模型不需要调整参数，就能从这些示例中“汲取”经验，处理类似的新任务

**基于交互经验的上下文学习 (Interaction-based ICL)**: 让机器人**自己通过与环境的短暂互动来收集上下文信息**

**基于检索的上下文学习 (Retrieval-based ICL)**: 这类方法为VLA配备了一个外部记忆库。在推理时，模型会根据当前场景，从这个记忆库中检索出最相关的过往经验作为上下文

Qwen-RoboManip: https://www.alphaxiv.org/pdf/2606.17846v2
```
**把历史执行轨迹编码成 token，和当前图像、语言指令一起送入 VLM；训练时随机采样历史窗口，防止模型复制最近动作；部署时使用最近历史的 rolling window，从而在不更新参数的情况下，根据机器人实际执行行为动态调整策略。**
```

LingBot-VA2.0:https://www.alphaxiv.org/pdf/2607.08639v2
训练阶段分为tokenizer training和video-action policy training
```
ICL数据构造：
1. 从多个机器人视频数据集按任务语义采样机器人轨迹；
2. 使用 VLM 分析任务，并生成把机器人第一帧改造成“人类操作场景”的编辑提示；
3. 编辑第一帧，得到人类第一视角操作的初始图像；
4. 再用视频生成模型生成对应的人类操作视频；
5. 用 VLM 对生成视频进行**任务语义保持度**和**物理合理性**评分；
6. 筛选合格的人类视频，与原始机器人轨迹配对，形成 ICL 样本
   
ICL train:
一个 ICL 样本包含：
- 当前机器人观测和历史；
- 机器人动作序列；
- 一段语义对应的人类演示视频。
  
人类视频 -> tokenized to a ICL latent: z_icl， 随后每个chunk都包含这个条件

ICL inference:
- 输入当前机器人观测；
- 输入一段人类参考视频作为任务示范；
- 不进行梯度更新或参数微调；
- 直接根据视频中展示的操作程序，在新的物体、场景或任务组合上生成机器人动作。
```


#### Casual video-action modeling:

robot trajectory is trunked into $\tau\{(x^i, a^i)\}_{i=1}^{N}$ 
where $x_{i}$ is the video chunk and $a_i$ is the temporally aligned action chunk.

target:
$$p_\theta(x^{i+1}, a^{i+1} | x^{\leq i}, a^{\leq i}, c)$$
joint prediction is factorized into video prediction followed by action decoding:
$$p_\theta(x^{i+1}, a^{i+1} | x^{\leq i}, a^{\leq i}, c) = p_\theta^{vid}(x^{i+1}| x^{\leq i}, a^{\leq i}, c)*p_\theta^{act}(a^{i+1} | x^{\leq i}, a^{\leq i}, c, x^{i+1}) $$


