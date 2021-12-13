## Acknowledgements

This project is for the fourth project of **COMP551** at **McGill University** in **fall 2021**. Here we bid thanks
to **Yuyan Chen**, **Ing Tian**, and **Zijun Zhao**, without whom this project cannot come real.

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

`acon.py` contains the optimizer `ACON` we wish to investigate in this experiment. Inside the `classifier` folder, we
have defined various models spanning from *VGG*, *AlexNet*, *ShuffleNet*, and *ResNet*. Also, some common utils
pertaining to these models are defined in `classifier/__init__.py`, `classifier/metric.py`, and `classifier/plugin.py`.
Next, the `data` folder contains utils related to dataset processing. Finally. we have provided a sample `p4.ipynb` to
run our codes in Colab.

---
