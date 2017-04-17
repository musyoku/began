## Boundary Equilibrium GAN

- Code for the [paper](https://arxiv.org/abs/1703.10717)
- [実装について](http://musyoku.github.io/2017/04/16/Boundary-Equilibrium-Generative-Adversarial-Networks/)

### Requirements

- Chainer

## Mixture of Gaussians Dataset

![image](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2017-04-16/gaussian.png?raw=true)

## Animeface Dataset

### Generator output (96x96)

![image](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2017-04-16/gen_output.png?raw=true)

### Interpolation of Generator output

![image](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2017-04-16/analogy.png?raw=true)

pretrained model:

https://drive.google.com/open?id=0ByQaxyG1S5JRc28wMDNjMWp0NTA

```
├── train_animeface
    ├── model
    |   ├── discriminator.hdf5
    |   ├── generator.hdf5
    |   └── model.json
    ├── interpolate.py
    └── train.py
```

### Mode collapse

![image](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2017-04-16/mode_collapse_2.png?raw=true)
