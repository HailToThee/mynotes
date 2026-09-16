对于一个lingbot va:
>action distributions are inherently simpler than visual data requiring fewer parameters to model effectively while maintaining expressive capacity for visual dynamics

autoaggressive proccedure:
		ot -> o_t+1 -> at
if we dont need a o_t+1, then we need a momentum
icl directly from ot->at(need explicit data collection of actions)
momentum is good?
how to define action

is there two hypersphere of video generation and action generation transformer?
how will the x^<=i move to x^<=i + x^i+1 on the hypersphere of action transformer?

Flow matching is used for velocity, which is otherwise called a momentum.
if the momentum is good enough, we need no longer more steps
seen types of dual denoising paradigms

kaiminghe's diffusion?

	
