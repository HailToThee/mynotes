background:
```
Policy under Zero-shot cross task generalization must execute manipulation it never seen during training.
In large language models, a novel task can be performed simply by specifying it in the context, without any parameter.This form of in-context learning (ICL) turns generalization into a problem of task specification.

two-key challenges:
1. large-scale, task-rich humanrobot paired data remain scarce and expensive to collect manually.
2. a policy trained on seen tasks can learn shortcuts that allow it to ignore the in-context video.(ICL中的human video先验可能并没有被使用，因为模型直接走shortcut就够了，导致并没有很好地利用信息)
```
![[zerowam_arc.png]]
*The main architecture follows lingbot-va*

Solution for 2 challenges:
> proposed a n **in-context human video generation pipeline**, uses task-sampled robot trajectories to construct semantically matched human manipulation videos.
  The resulting HumanGen dataset contains 74.2K human-robot ICL pairs spanning 8.6K tasks.

```
Robot Trajectory
    ↓
VLM Task Analysis
- task name
- initial object states
- object state changes
- final object states
    ↓
Generate Image-Editing Prompt
    ↓
Image Editing Model
Robot First Frame → Human Initial Frame
    ↓
VLM Generates Video Prompt
based on task-level object state transitions
    ↓
Video Generation Model
Human Initial Frame + Video Prompt
→ Human Demonstration Video
    ↓
VLM Quality Filtering
- task semantic consistency
- physical plausibility
    ↓
Human–Robot ICL Pair
(Human Demo Video, Robot Trajectory)
```
也就是说需要手动合成数据

> introduce an in-context future chunk prediction (IFP) objective.
> (标准的next-chunk prediction仅仅使用当前的state和instruction就可以生成，因此常常忽略contexts)
> supervises multiple strided chunks of future robot video from the current robot-video representation, encouraging the model to encode the longer-term task evolution conveyed by the human video.

**In-Context Future Chunk Prediction:**
对于每一个chunk， predict k-th future target video chunk: $x^{j_k}$

$$j^{k}= (i+1) +1 +(k-1) s, \quad k = 1, 2, .., K.$$
如何把这个融合到模型的参数里面？
用一个单独的MLP， 进行flow matching的匹配，作为一个损失项即可，特别的，论文用了在main stream预测下一个**chunk $x_{i+1}$  under $C^{vid, i}$** 观察在transformer中各个层的输出(layer = M)，concat之后作为一个条件：
$$\phi^{i+1} = P_{fuse}(Concat(\{r_m^{i+1}\}_{m=1}^{M}))$$
假设现在集成了K个不同的MLP作为IFP module, 每个单独预测一个
$$p^{ifp}_{\theta , k} (x^{j_{k}}|\phi^{i+1}, x^{\leq i}, a^{\leq a}, l)$$
Loss即我们提到的flow matching:
$$\mathcal{L}_{ifp} = \sum\limits_{k = 1}^{K}w_k\mathcal{L}_{fm}(x^{j_{k}}|\phi^{i+1}, x^{\leq i}, a^{\leq i}, l)$$

**Inference:**
推理的时候，我们并不需要预测k个step，所以刚刚引入的IFP module可以卸载掉。
而训练的时候已将上述的human video信息引入到next chunk prediction中，实现了想要的结果。


**human video作用在哪里？**
![[zerowam_condition.png]]
上面也说了under C
也就是说h作为一个整体，先作用于next chunk，然后再从next chunk作用到action
这里面的公式，我们可以看到对于action部分，h并不直接作为其条件，而x_i+1作为其条件