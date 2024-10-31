# Installation
Our installation follow the installation of [OpenFlamingo](https://github.com/mlfoundations/open_flamingo). 

If you want to evaluate in-context classification performance on **OpenFlamingo-9B-vitl-mpt7b** ，the version of transformers is `4.33.2`.

If you want to evaluate in-context classification performance on **idefics2-8b-base**，then you need to update transformers：

```
pip install --upgrade transformers
```

# Usage

### 1.cache the image feature of dataset

```
cd ImageNet
python extract_data_feature.py
```

### 2.Get the results on ImageNet

```
cd ImageNet
python main.py --device {device} --model {model_name} --method {method}
```