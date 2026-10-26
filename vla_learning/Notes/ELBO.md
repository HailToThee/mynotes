For VAE(Variational AutoEncoder) like image generation models. ELBO is used to transform untactible logit likelihood to lower bound that can be optimized.

![[VAE_arc.png]]
let $q_\phi$ denotes the encoder and $p_\theta$ denotes the decoder.
$x$ for input, $z$ for latent space and $x'$ for reconstructed image(output).
The goal is:
$$\min \mathcal{D}_{KL}(q(z|x), p(z|x))=\sum_{x} q(z|x)\log \frac{q(z|x)}{p(z|x)}$$
