# Video VAE / Joint Flow Matching / Long-WAM 笔记

### 1. Video VAE

给定原始视频 $V\in\mathbb R^{B\times C\times T\times H\times W}$，Causal Video VAE encode：$Z_0=E_{\mathrm{VAE}}(V)$

Compressed latent representation: $Z_0\in\mathbb R^{B\times C_z\times T'\times H'\times W'}$

将其 Patchify 并展平为时空 Token 序列：$X_0=[x_1,x_2,\ldots,x_N]$

* Causal VAE is used to constrain the temporality in **encoder**

### 2. Joint Flow Matching

### Noised latents

对整段视频的潜在表示采样高斯噪声 $\epsilon\sim\mathcal N(0,I)$ 和噪声水平 $\sigma\in[0,1]$。
采用线性插值：

$$
X_\sigma=(1-\sigma)X_0+\sigma\epsilon
$$

velocity field:

$$
u_\sigma=\frac{dX_\sigma}{d\sigma}=\epsilon-X_0
$$

### Bidirectional Self-Attention

整段视频的所有时空 Tokens 同时输入 DiT。

对于某一层 Self-Attention：$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$

$$
\operatorname{Softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

由于不使用时间上的 Causal Mask，任意 Token $x_i$ 都可以访问其他时间位置的带噪 Tokens。

因此，预测第 $i$ 个 Token 的速度时：$\hat u_i=v_{\theta,i}(X_\sigma,\sigma,c)$，其输出可以依赖整个 $X$。

### training target

DiT 联合预测整段视频的速度场：$\hat U=v_\theta(X_\sigma,\sigma,c)$

Training loss：

$$
\mathbb E_{X_0,\epsilon,\sigma}
\left[
\left|
v_\theta(X_\sigma,\sigma,c)-(\epsilon-X_0)
\right|_2^2
\right]
$$

所有时空 Tokens 共同参与损失计算。

**Joint 的含义：** 所有时间位置通过 Bidirectional Attention 发生信息交互，形成联合速度场，并在同一次去噪过程中更新。

### 2.4 推理：整段视频联合去噪

从高斯噪声开始：$X_{\sigma_0}\sim\mathcal N(0,I),\qquad \sigma_0=1$

通过数值积分逐步降低噪声水平：

$$
X_{\sigma_{k+1}}
=
X_{\sigma_k}
+
(\sigma_{k+1}-\sigma_k)
v_\theta(X_{\sigma_k},\sigma_k,c)
$$

其中 $\sigma_{k+1}<\sigma_k$。

每一次更新的对象都是整段视频，而非单独某一帧。

最终得到 $X_{\sigma=0}$，再通过 VAE Decoder 重建视频。

## 3. Long-WAM：Autoregressive Flow Matching

Long-WAM 使用 Block-Causal Attention，将视频划分为多个时间 Chunk：$Z=[Z_1,Z_2,\ldots,Z_K]$

### 3.1 自回归视频建模

视频的概率分布按照时间块分解：

$$
\prod_{i=1}^{K}
p_\theta(Z_i\mid Z_{<i},c)
$$

其中 $Z_{<i}=[Z_1,\ldots,Z_{i-1}]$ 是此前的历史视频块。

预测当前 Chunk 时，可以访问此前的历史以及当前 Chunk 内的带噪 Tokens，但不能访问后续真实视频块。

### 3.2 Chunk-level Flow Matching

对于当前目标 Chunk $Z_i$，构造：

$$
Z_i^\sigma=(1-\sigma)Z_i+\sigma\epsilon_i
$$

对应的目标速度：

$$
u_i=\epsilon_i-Z_i
$$

视频模型预测：

$$
\hat u_i=v_\theta(Z_i^\sigma,\sigma\mid Z_{<i},c)
$$

训练目标为：

$$
\mathbb E
\left[
\left|
v_\theta(Z_i^\sigma,\sigma\mid Z_{<i},c)
-
(\epsilon_i-Z_i)
\right|_2^2
\right]
$$

这是基础形式。Long-WAM 实际的 AR 预训练还采用 Teacher Forcing、Error Recycling 和 Block-Causal Attention，部分目标与条件会受到扰动。

### 3.3 Teacher Forcing

训练时，使用 Ground-truth History $Z_{<i}$ 作为当前 Chunk 的条件：

$$
v_\theta(Z_i^\sigma,\sigma\mid Z_{<i}^{\mathrm{GT}},c)
$$

不同 Chunk 可以在 Block-Causal Mask 下并行计算训练损失。

推理时，则使用已经观测或生成的历史：

$$
\operatorname{FM-Sample}_\theta
(\epsilon_i\mid Z_{<i},c)
$$

生成当前 Chunk 后，将其加入历史，再生成下一个 Chunk。

因此：

- **训练阶段：** 利用 Teacher Forcing 并行监督多个 Chunk。
- **推理阶段：** 沿时间方向依次生成 Chunk，每个 Chunk 内进行 Flow Matching。

