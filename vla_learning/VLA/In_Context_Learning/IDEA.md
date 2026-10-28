	
以 Zero-WAM / LingBot-VA 为例：

$$
\text{Demo / Instruction} + V_{\le t}
\rightarrow
\hat V_{t+1}
\rightarrow
A_t
$$

即：
1. 根据 Human Demo、Instruction 和当前 Robot Observation 预测 Future Robot Video；
2. 根据 Future Visual State 预测 Robot Action。

Fast_WAM 提出了不必完整地预测整个 $V_{t+1}$, OpenWAM也讲到了这一点
因为生成 $V_{t+1}$ 需要建模大量可能与 Action 无关的细节信息，例如：
	- Background
- Texture
- Appearance details
- Pixel-level details

需要的信息可能只是：

> **当前 Visual State 接下来应该朝什么方向变化？**


$$
z_t
\rightarrow
\hat z_{t+1}
\rightarrow
A_t
$$

不显式生成 $\hat z_{t+1}$，而是预测从当前状态指向 Future State 的方向：
通过
$$
V_t
\rightarrow
d_t
\rightarrow
A_t
$$


如何得到dt?
```
先使用最简单的 Latent Displacement：

d_t = z_{t+1} - z_t
	
训练 Visual Dynamics Predictor：

\hat d_t = D_\phi(z_t,H)

minimize MSE loss or KL divergence
```

```
上述的可能存在一个问题，即dt仅凭借z_t z_t+1能否对应正确的time step？
对于d_t来说，应该是通过前面所有的o_t得到的，因此需要维护一个状态量s_t
s_t = mamba(z_1:t)
由(H, s_t, z_t+1)  --> d_t
```
Training

训练阶段仍然可以保留 Future Prediction 作为 Auxiliary Supervision, 因为仍然可能走捷径
看看使用预测出来的方向能否指导恢复？
Future Loss：

$$
L_{\mathrm{future}}
=
d(\hat z_{t+1},z_{t+1})
$$

结构：

```text
Human Demo H
      +
Current Robot Video V_t
      ↓
Visual Dynamics Predictor
      ↓
    d_t
   /   \
  /     \
 ↓       ↓
V_t+1   Action
 ↓       ↓
L_future L_action
```

Inference

推理时删除 Future Reconstruction Branch， 不显式地生成V_t+1


>human video In context learning的作用？

由于human video和robot video之间存在gap, 因此不能直接 human video --> action
而是需要加一个连接使得
(h, v_t) --> dt
(v_t, dt) --> At

如何设计？
1. 整体直接输入到input端，然后crossattn得到一个visual latent
2. 可以参照Bahavior Prompting Policy？也是根据当前的o_t以及动作状态，取argmax(attn(cur, video chunks))得到当前对应的human帧的chunk?再根据这个chunk和下一个chunk作为input，进行输入来指导dt的生成？然而存在问题，第一点是数据问题，Bahavior Prompting Policy的训练数据中有直接对应的当前动作。第二点是embodiment gap的问题，这个能否在训练中缓解？
3. 如果使用了状态量s_t， 对于human demo，能否也通过mamba编码成N个embedding，再使用bahavior prompting policy进行状态的匹配？这个是否会没有那么大的embodiment gap
df

如何包含past的trajectory?
from zerowam... bahavior prompting policy, 如果不包含trajectory的话，当前帧能够提供fairly多的信息，因此未必会考虑添加的额外部分。也就是直接input进去作为context可能并不work

如何把这个trajectory输入到context里面？强行关注trajectory信息。
with post training or without post training
if with post training

1. follow zerowam等通过添加对未来帧的预测mse loss显式将video context引入进来，和上面介绍的类似
2. 实现一种对context压缩的方法，通过post training让模型能够理解包含trajectory等信息。
3. geometric transfer, trajectory may also be presented  in hyper latent space? 相关论文的调研
2,3 很可能可以融合

long-wam
in context learning for robots调研1
记忆和incontext十分类似




1.Mamba...

1. 使用预训练 Wan VAE 将 Human Demonstration Video 编码为 Visual Tokens，再通过 Mamba 建模时序信息，取最后一个 Visual Token 的 Hidden State 作为 Context Latent，输入 VLA 指导动作生成。
    
2. Mamba 需要学习如何保留任务相关的时序信息，可能需要额外 Pretraining，与 Minimal Training 的目标存在冲突。
    
3. 能否不使用单个 Token，而是将 Video 压缩为 \(N\) 个具有时序信息的 Embeddings，再根据当前 Robot Observation \(o_t\)，通过 Attention 匹配对应的 Human Demonstration 阶段，提取相关 Context 指导 Action Generation？
![[aem(mamba)-arc.png]]

![[aem(mamba)-algor.png]]


2.latent 

如何设计？

1. 分别使用预训练 Wan VAE 和 DINOv3 对 Human Demonstration Video 进行编码，得到每一帧对应的 Latent，观察其在 Latent Space 中的演化轨迹。
    
2. 由于 Wan 和 DINO 的特征都可以进行归一化，能否比较两者的 Latent Trajectory 是否具有相似的几何性质？例如相邻帧变化方向、轨迹曲率、局部切空间以及状态转移趋势。
    
3. 如果两种不同预训练目标的模型得到相似的 Latent Trajectory，是否意味着其中存在相对稳定的任务动态信息？能否利用这种动态结构提取有效的 Video Context，而不需要额外预训练？
    
4. 进一步比较相同任务下 Human Video 和 Robot Video 的 Latent Trajectory，观察其运动方向、局部几何结构是否相似，探索能否通过共享的动态结构缓解 Embodiment Gap。