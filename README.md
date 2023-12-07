# Generative-Adversarial-Networks

The adversarial modeling framework is most straightforward to apply when the models are both
multilayer perceptrons. To learn the generator’s distribution pg over data x, we define a prior on
input noise variables pz (z), then represent a mapping to data space as G(z; θg ), where G is a
differentiable function represented by a multilayer perceptron with parameters θg . We also define a
second multilayer perceptron D(x; θd ) that outputs a single scalar. D(x) represents the probability
that x came from the data rather than pg . We train D to maximize the probability of assigning the
correct label to both training examples and samples from G. We simultaneously train G to minimize log(1 − D(G(z)))

In other words, D and G play the following two-player minimax game with value function V(G, D)


       min max V (D,G) = E(x∼pdata(x))[log D(x)] + E(z∼pz(z))[log(1 − D(G(z)))].
       

ALgorithm for GAN:

![image](https://github.com/prabandh1444/Generative-Adversarial-Networks/assets/111416767/dd83dd79-2c58-4d0d-a1fd-f73081839a5f)

Results on various latent variables z over gaussian Distribution on MNIST dataset:

![image](https://github.com/prabandh1444/Generative-Adversarial-Networks/assets/111416767/eb44d084-885d-4b34-8e65-14715e3420b8)

# Cyclic Generative-Adversarial-Networks
Our goal is to learn mapping functions between two
domains X and Y given training samples {xi }N
i=1 where 1 xi ∈ X and {yj }M j=1 where yj ∈ Y . We denote the data
distribution as x ∼ pdata (x) and y ∼ pdata (y). As illus-
trated in Figure 3 (a), our model includes two mappings
G : X → Y and F : Y → X. In addition, we in-
troduce two adversarial discriminators DX and DY , where
DX aims to distinguish between images {x} and translated
images {F (y)}; in the same way, DY aims to discriminate
between {y} and {G(x)}

<pre>
       LGAN(G,DY,X,Y) = Ey∼pdata (y) [log DY (y)] + Ex∼pdata (x) [log(1 − DY (G(x))]
</pre>
We introduce a similar adversarial loss for the mapping function F : Y → X and its discriminator DX as well:
i.e., minF maxDX LGAN (F, DX , Y, X).

Adversarial training can, in theory, learn mappings G
and F that produce outputs identically distributed as target
domains Y and X respectively (strictly speaking, this re-
quires G and F to be stochastic functions) [15]. However,
with large enough capacity, a network can map the same
set of input images to any random permutation of images in
the target domain, where any of the learned mappings can
induce an output distribution that matches the target dis-
tribution. Thus, adversarial losses alone cannot guarantee
that the learned function can map an individual input xi to
a desired output yi . To further reduce the space of possi-
ble mapping functions, we argue that the learned mapping
functions should be cycle-consistent.

<pre>
       Lcyc (G, F ) = Ex∼pdata (x) [||F (G(x)) − xk||] + Ey∼pdata (y) [||G(F (y)) − y||].
</pre>

Our full objective is:
<pre>
G*,F* = min max V (D,G) LGAN (G,DY,X,Y)+ LGAN (F,DX,Y,X)+ λLcyc (G,F)
</pre>

We use this model to transfer the style of great painter(monet) into camera photos:
A are the set of cam photos while B are drawn by monet


![image](https://github.com/prabandh1444/Generative-Adversarial-Networks/assets/111416767/7b43c9fe-236d-43b7-b1e3-cea2c9da2fec)

![image](https://github.com/prabandh1444/Generative-Adversarial-Networks/assets/111416767/4c4ff720-425e-4c80-bcb6-c15f88356fc1)


