## Fast-WAM: Do World Action Models Need Test-time Future Imagination?
```
background:

Most exsiting WAMs follow an imagine-then-execute paradium: first generate future observations, then predict actions conditioned on the imagined future.

Why WAMs are effective prediction future physical staets?

The paper thinks that there may stem from 2 distinct resources:
(1) the video prediction objective during training, which may help the model acquire stronger physical priors and action-conditioned representations, and 
(2) explicit future generation during inference, which may provide additional foresight for action prediction. 
\mathcal{L}_{act} = \mathcal{L}_{FM}(a_{1:H}), 
Existing WAM systems typically entangle these two factors, making it difficult to determine which one is actually responsible for the observed gains.

question: is it necessary to predict future image during inference?

```
![[representative_wam_paradigms.png]]![[fastwam_arc.png]]```
```
Model Structure:
During inference, Fast-WAM does not explicitly generate future observations. Instead, it keeps only the clean latent tokens of the first observation frame, processes them with the video model in a single forward pass, and uses the resulting latent world representation for direct action generation.
```
Training objective:
Like LingBot:
$$y_{t}= (1-t)y+t\epsilon$$
$$\mathcal{L}_{FM}(y)=\mathbb{E}_{y, \epsilon, t}[\|f_{\theta}(y_{t}, t, o, l)-(\epsilon-y)\|_2^2]$$
$$\mathcal{L}_{act} = \mathcal{L}_{FM}(a_{1:H}), \mathcal{L}_{vid} = \mathcal{L}_{FM}(z_{1:H}) $$
overall
$$\mathcal{L} = \mathcal{L}_{act}+\lambda\mathcal{L}_{vid} $$Results:
![[fastwam_res1.png]]
![[fastwam_res2.png]]![[fastwam_res3.png]]


## World Action Models are Zero-shot Policies
```
background:

While VLAs successfully inherit linguistic priors to generalize across diverse language instructions, especially manipulating diverse objects, their generalization to novel environments and, more critically, to new motions or skills remains limited

**WAMs (World Action Models)**: Models that jointly predict video and action, leveraging rich spatiotemporal priors from pretrained video diffusion backbones.

```

**Why WAMs are more effective? Their point of view:**

> "WAMs learn physical dynamics by predicting future world states and actions, using video as a dense representation of how the world evolves."

> "This shifts action learning from dense state-action imitation to inverse dynamics—aligning motor commands with predicted visual futures."

![[dreamzero_arc.png]]
**Model Arch:**

$$
\underbrace{\pi_\theta(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{DreamZero}}
=
\underbrace{\pi_\theta(\mathbf{o}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{video prediction}}
\cdot
\underbrace{\pi_\theta(\mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l+H}, \mathbf{q}_l)}_{\text{IDM}}

$$
> "Instead of using two separate models (video prediction model and inverse dynamics model) to model the decomposed objective, we train a single model end-to-end with joint prediction objective."
#### Backbone
- **14B autoregressive DiT**, initialized from Wan2.1-I2V-14B-480P (pretrained video diffusion model)
- Uses **flow matching** as training objective
- **Chunk-wise generation**: each chunk has K=2 latent frames, action horizon H=48 steps (1.6 seconds)
- **Teacher forcing**: model denoises the noisy current chunk conditioned on clean previous chunks
#### Inference
- Uses **KV caching** for efficient inference
- After each action chunk executes, **ground-truth observations replace predicted frames** in the KV cache
- Eliminates compounding error accumulation
- Asynchronous execution: inference runs concurrently with action execution
### Training Objective

Given chunk index \(k > 0\) and denoising timestep $t_k \in [0,1]$:

$$\mathbf{z}_{t_k}^k = t_k\mathbf{z}_1^k + (1-t_k)\mathbf{z}_0^k,\quad
\mathbf{a}_{t_k}^k = t_k\mathbf{a}_1^k + (1-t_k)\mathbf{a}_0^k$$
$$C_k = \{(\mathbf{z}_1^j, \mathbf{a}_1^j)\}_{j=1}^{k-1}
$$

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z},\mathbf{a},\{t_k\}}\left[\frac{1}{K}\sum_{k=1}^{K}w(t_k)\| \mathbf{u}_\theta([\mathbf{z}_{t_k}^k,\mathbf{a}_{t_k}^k];C_k,\mathbf{c},\mathbf{q}_k,t_k) - \mathbf{v}^k\|^2\right]
$$


$$\mathbf{v}^k = [\mathbf{z}_1^k, \mathbf{a}_1^k] - [\mathbf{z}_0^k, \mathbf{a}_0^k]
$$

> "DreamZero shares the denoising timestep between video and action modality for faster convergence at the beginning of training."

### Inference Optimization: DreamZero-Flash(for faster)

**Problem**: Standard training uses shared timestep $t_k \sim \mathcal{U}(0,1)$ for both modalities. With few-step inference (≤4 steps), actions need to denoise quickly while video remains partially noisy—creating a **train-test mismatch**.

**Solution: Decoupled noise schedules**
$$
t_k^{\text{video}} = 1 - \eta,\quad \eta \sim \text{Beta}(7,1),\quad t_k^{\text{action}} \sim \mathcal{U}(0,1)
$$

> "DreamZero-Flash closes this gap by biasing video timesteps toward high-noise states... while action timesteps remain uniform. be faster and faster"

**Effect**:
	4 denoising steps → 1 step
	Inference latency: 350ms → 150ms (2.33× speedup). Task progress: 52% (standard, 1-step) → 74% (Flash, 1-step), only 9% below 4-step baseline (83%)
![[dreamzero_algor.png]]