## Acknowledgements

This project is for the fourth project of **COMP551** at **McGill University** in **fall 2021**. Here we bid thanks
to **Yuyan Chen**, **Ing Tian**, and **Zijun Zhao**, without whom this project cannot come real.

---

## Brief

In this project, we investigate the effects of a family of activation functions `ACON` proposed in
this [paper](https://arxiv.org/abs/2009.04759). Concretely, we explore the effects of `ACON` and `Meta-ACON` with
respect to `ReLU` in various experimental setups. For example, we have tested the performance of `ACON`, `Meta ACON`,
and `ReLU` for variants of `VGG16` on the `CIFAR-100` datset.

---

## Project Structure

```text
.
├── README.md
├── acon.py
├── classifier
│   ├── __init__.py
│   ├── metric.py
│   ├── network
│   │   ├── __init__.py
│   │   ├── alex_net.py
│   │   ├── resnet_acon.py
│   │   ├── resnet_metaacon.py
│   │   ├── resnet_relu.py
│   │   ├── shuffle.py
│   │   ├── shuffle_acon.py
│   │   ├── shuffle_metaacon.py
│   │   ├── vgg16_6_acon.py
│   │   ├── vgg16_6_metaacon.py
│   │   ├── vgg16_6_relu.py
│   │   ├── vgg16_acon.py
│   │   ├── vgg16_metaacon.py
│   │   └── vgg16_relu.py
│   └── plugin.py
├── data
│   └── __init__.py
├── main.py
└── p4.ipynb
```

`acon.py` contains the activation functions `ACON` and `MetaACON` we wish to investigate in this experiment. Inside the `classifier` folder, we
have defined various models spanning from *VGG*, *AlexNet*, *ShuffleNet*, and *ResNet*. Also, some common utils
pertaining to these models are defined in `classifier/__init__.py`, `classifier/metric.py`, and `classifier/plugin.py`.
Next, the `data` folder contains utils related to dataset processing. Finally. we have provided a sample `p4.ipynb` to
run our codes in Colab.

---

## How to run codes in Colab?

Though Colab is convenient, we suffer from frequent disconnections. Hence, we often run our script locally via `main.py`
. For your convenience, we have provided a sample `ipynb` file such that you can replicate our results in Colab. The
procedure is simple.

1. Name the project folder as `COMP551_P4`.
2. Zip the project folder into `COMP551_P4.zip`.
3. Open the `p4.ipynb` in Colab and connect to a **GPU** runtime.
4. Upload `COMP551_P4.zip` to the Colab runtime under the `/content` folder.
5. Run the Jupyter notebook.
