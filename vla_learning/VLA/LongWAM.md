https://arxiv.org/pdf/2610.10528
![[longwam.png]]

如何有效地利用历史信息？
Autoreggressive video pretraining.(Longlive)

Teacher forcing training: 将预测目标分解为chunk by chunk, 并行进行训练, 每个只能chunk看到自己之前的chunks. 
另外一个问题，并行训练时候如果只使用ground truth进行训练，误差累计并没有被考虑，在inference的过程中可能出现问题
沿用longlive-2.0的Error Recycling
使用buffer记录误差，然后在iteration中进行迭代
![[longlive-error.png]]


先前的工作存在一个问题，即video pretrained model are bidirecticlly trained而下游的video and action predict则需要causal prediction, 这样就不太对齐？
![[longwam-rollout.png]]
类似于lingbotva2.0

对于inference speed
采用了一个异步VAE架构
![[longwam-streamingVAE.png]]

### longlive
![[longlive_arc.png]]

- Using KV recache for user prompt injection.(Order instruction)
- Streaming long tuning (For coherence)
- ![[longlive-streaming.png]]
- Short window attention+frame sink.
- ![[longlive-framesink.png]]


changxiawen