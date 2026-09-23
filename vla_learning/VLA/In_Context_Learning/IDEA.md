
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
