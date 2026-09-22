World–Action Models inherit world knowledge from video-generative priors, and channel it into executable control signals through embodied experience.
```
先前的系统是整体的：
the generative backbone, visual representation, architecture, information flow, inference procedure, and training data are tightly coupled, obscuring which design choices matter and why.
没法搞清楚哪个设计有用，为什么有用
```
![[openwam_arc_of_ablation.png|700]]

尝试回答三个问题：
>1. What world knowledge should a WAM inherit?
>结论：上游的world knowledge通过一个足够强大的backbone和一个紧凑，信息丰富的visual latent最有效地转移
![[openwam_res3.png|203]]![[openwam_res4.png|432]]
>2. How can we create synergy between world and action learning?
>结论： It is best fostered with sufficient action-specific capacity, explicit world-to-action information flow, and a joint test-time denoising schedule.
>**action-specific capacity:**  action部分需要一个自己的DiT，
>![[openwam_res2.png|610]]
>**explicit world-to-action information flow:** visual和action部分的信息可以分类为
>```
>isolated
>action sees visual
>visual sees action
>mutual  这四种
>```
对这四种分别进行测试，发现action sees visual效果很好，而反过来则没有同样的收益。所以action必需能够读取world representation。进一步，这里的world representation代表什么？不一定必须是完整的 $o_{t+1}$ , leaving spaces for ICL
![[openwam_res1.png]]
**joint test-time denoising schedule:** video latent和action latent到底以什么样的顺序denoise到clean？也是分为好几种，结果是二者的time step remain same的同时，video -> video -> action -> aciton这样的效果比较好
![[openwam_rse6.png|695]]

>3. How can this synergy be scaled across domains?
>



#### openwam-$\alpha$ 
![[openwam_arc.png]]