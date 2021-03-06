# hcl-on-fashion-mnist

### Intro

This is fun project to experiment with Hard Contrastive Learning on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

### Solution strategy
In order minimize the amount of labeled data, we use an unsupervised ML model. The model, trained in an
unsupervised fashion, is then finetuned (in a supervised way) on labeled samples in order to reach >90% on the test set.

There is a lot of work on unsupervised methods in the literature, we have chosen to follow [this paper](https://arxiv.org/pdf/2010.04592.pdf) starting from their code.

The training dataset (60000 images) is split into:
* 55000 unlabeled images used for training
* 4000 labeled images used for training
* 1000 labeled images used for validation

### Requirements
* Python >=3.9 (previous versions might work but is not guaranteed)
* at least one GPU (the code run anyways without GPU, but it takes very long)

### Installation
Clone the repository
```bash
$ git clone git@github.com:giaco5988/hcl-on-fashion-mnist.git
```
Enter the repository, create a virtual environment and install required packages
```bash
$ cd hcl-on-fashion-mnist
$ python3.9 -m venv venv_hcl
$ source venv_hcl/bin/activate
(venv_hcl) $ python
(venv_hcl) $ pip install -r requirements.txt
```

### Training
First, let's train the unsupervised model (it takes ~10hours with 1 GPU)
```bash
(venv_hcl) $ python main.py train_hcl
```
Second, let's finetune on the labeled data
```bash
(venv_hcl) $ python main.py finetune pretrained_path=path/to/pretrained/model
```

NOTE: results are saved on `lightning_logs` folder.

### Results: test set accuracy 90% using 5000 labeled images

Below, see results for training loss and validation accuracy for the **unsupervised** model

![](https://github.com/giaco5988/hcl-on-fashion-mnist/blob/main/docs/Screenshot%20from%202021-10-05%2023-36-12.png)
![](https://github.com/giaco5988/hcl-on-fashion-mnist/blob/main/docs/Screenshot%20from%202021-10-05%2023-36-21.png)

Below, see results for training loss and validation accuracy for the **supervised** model

![](https://github.com/giaco5988/hcl-on-fashion-mnist/blob/main/docs/Screenshot%20from%202021-10-05%2023-37-40.png)
![](https://github.com/giaco5988/hcl-on-fashion-mnist/blob/main/docs/Screenshot%20from%202021-10-05%2023-37-34.png)

### Next Steps
* Train for longer time
* Add callbacks to plot inference on one batch during training
* Try different models for the base model
* Hyperparameter search, use different training and model parameters (e.g. lr, batch size, feature dimension)
* Add tests
* Add dvc-tracking to track data and pipelines
* In order to reduce queries to database, can we understand which are the most effective (optimal) labels to query?
E.g. those which are most difficult to classify. A first try could be to query those with low classification score.
This is similar to what is done in active learning.
