## BagelVLA: Enhancing Long-Horizon Manipulation via  Interleaved Vision-Language-Action Generation
```
background

Some methods focus on high-level planning  but lack visual forecasting, while others focus on visual prediction but struggle with the logical reasoning required for complex tasks. A unified framework that seamlessly integrates reasoning, prediction, and control remains a key challenge.

现有 VLA 把 reasoning、prediction、control 分开了，因此在 long-horizon manipulation 中不够好

jointly process and generate text and images, exhibiting emergent abilities in multimodal reasoning. 
These models provide an appealing prior for embodied agents: the model can “think” about the next step in text and “imagine” the outcome in pixels. 
However, such general-purpose models are not designed for embodied domain reasoning and continuous real-time control.
```
**为什么同时做language planning and Visual forecasting?**
对于long-horizon tasks:
where a global instruction (e.g., stacking blocks in a specified order (red→yellow→blue→green)) implicitly entails a sequence of distinct stages. We address this by modeling the problem as **Interleaved Planning**. Instead of a black-box mapping, we require the model to explicitly reason through the causal chain of the task.

也就是说不能简单的学习
>observation + global instruction + action
而是应该学习
>language + future visual state + action (Interleaved Planning)


因此把三个目标: linguistic reasoning, visual forecasting, and action generation 结合起来，形成一个统一的 VLA 模型。按照“语言+视觉+动作”的顺序生成
联合分布: $$p_{\theta}(a_{t,}v_{t+k}, l_t|v_{t,}L) = p_{\theta}(l_{t}|v_{t,}L)*p_\theta(v_{t+k}|l_{t,}v_{t,}L)*p_\theta(a_t|v_{t+k}, l_{t,}v_{t,}L)$$
![[bagelVLA_arc.png]]

**Inference Speed**
如果按照最一般的想法，那么会先生成future image, 再生成action, 但是这样会很慢。因为future image本身需要很多的denoising steps
![[Types_of_dual_denoising_schemes.png]]
Three interaction mechanisms for the Flow Matching (FM) of keyframe prediction and action generation

对于scheme3 来说是很重要的，即Action不一定需要完整地生成future image, 只需要第一步的visual denosing得到的 latent KV cahce 就可以指导 action 这和Fast WAM很像

可是如何使visual denosing尽可能快地指导action呢？ 也就是如何在denoising的过程中尽可能早地得到有用的latent KV cache？
论文提出了RFG的想法:
普通的**single-step**: $v_{t+k}^{\tau = 0} \sim N(0, I)$ 从Gaussian开始
**RFG**: $v_{t+k}^{\tau = 0} \sim N(v_t, I)$: current observation as inital visual prior.