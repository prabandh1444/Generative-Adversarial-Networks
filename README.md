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

Results on various latent variables z over gaussian Distribution:

![image](https://github.com/prabandh1444/Generative-Adversarial-Networks/assets/111416767/eb44d084-885d-4b34-8e65-14715e3420b8)

