training data splits and selection...(先low resolution --> high resolution for finetune)

### Wan-VAE arc

![[wanVAE.png]]

类似于传统VAE的架构，但是其中需要address几个问题:

![[wanVAE2.png]]其一：视频具有时序性，传统VAE只是用来生成图片的
其二：视频的多个frame和high resolution导致计算开销很大，导致很难让VAE自动地scale到长视频序列
其三：如果要生成连续一致的视频，时序性尤其重要，但这也引入了额外的开销

维持a sequence![[wanVAE3.png]]

视频迁移的调研，因为实际上human->robot和视频迁移是很像的，但是还有一个问题就是这个过程怎么变得更快。

如何在wan这种bidirect DiT arc也做一个类似于能够记忆的功能，
longwam把记忆的功能交给了longlive，其实是具体的大规模数据的自回归过程。
