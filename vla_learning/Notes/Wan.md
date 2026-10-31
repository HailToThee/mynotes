training data splits and selection...(先low resolution --> high resolution for finetune)

### Wan-VAE arc

![[wanVAE.png]]

类似于传统VAE的架构，但是其中需要address几个问题:

![[wanVAE2.png]]其一：视频具有时序性，传统VAE只是用来生成图片的
其二：视频的多个frame和high resolution导致计算开销很大，导致很难让VAE自动地scale到长视频序列
其三：如果要生成连续一致的视频，时序性尤其重要，但这也引入了额外的开销

![[wanVAE3.png]]

**时序性的引入**: 类似于Mamba的Causal 3d convolutional layer![[Pasted image 20261031165850.png]]
padding_principle
```python
nn.Conv3d: self.padding = (T, H, W)
class CausalConv3d(nn.Conv3d):
	def __init__(self, *args, **kwargs)
		super().__init__(*args, **kwargs)
		self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0,) # see padding principle
		self.padding = (0, 0, 0)
	def forward(self, x, cache_x = None):
		padding = list(self.padding)
		if cache_x is not None and self._padding[4] > 0:
			cache_x = cache_x.to(x.device)
			x = torch.cat([cache_x, x], dim=2)
			padding[4] -= cache_x.shape[2]
		x = F.pad(x, padding)
		return super().forward(x)
```

### Wan arc

在训练了Wan-VAE用于时序信息的encode之后
apply一个DiT用于信息的处理，再通过decoder进行预测
![[Pasted image 20261031170205.png]]在此过程中的形状变化:
```python
print(latent.shape)
# [B, 1+T/4, H/8, W/8, C]
to input DiT blocks, apply a 3D convolution with a kernel size of (1, 2, 2) and apply a flattenting operation.
print(input.shape)
# [B, (1+T/4) * H / 16 * W / 16, C]
#
sigma = torch.rand(2, 1, 1)
noise = torch.randn_like(input)
# flow matching:
nosisy = (1 - sigma) * input + sigma * noise
target_velocity = noise - input
pred_velocity = DiT_model(noisy, sigma[:, 0, 0])

loss = F.mse_loss(pred_velocity, target_velocity)
loss.backward()
optimizer.step() 
```
所以这里所谓的双向attention实际上是对一个[(1+T/4) * H / 16 * W / 16, C]进行self-attn
这意味着任意一个patch都能attn到整个视频

视频迁移的调研，因为实际上human->robot和视频迁移是很像的，但是还有一个问题就是这个过程怎么变得更快。

如何在wan这种bidirect DiT arc也做一个类似于能够记忆的功能，
longwam把记忆的功能交给了longlive，其实是具体的大规模数据的自回归过程。
