# HW Medical Generative Adversarial Nets (MedGAN)

## Overview

In this homework, you will get introduced to [MedGAN (medical generative adversarial networks)](http://proceedings.mlr.press/v68/choi17a/choi17a.pdf). MedGAN was proposed to learn from electronic healthcare records (EHRs) and then generate synthetic EHRs. The main reason to do so is to circumvent private issues when sharing sensitive medical records to the public. Take our course for example, we synthesize MIMIC-III data for supporting our course coding assignments. The original version of MedGAN only does *unconditional* generation thus is unable to generate patient records with the specific desired property. In this homework, we will make a slight adaption of MedGAN to form a *conditional* MedGAN that is able to generate patient EHRs who are probably diagnosed with heart failure.


## About Raw Data

We will use a dataset synthesized from [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/).

The data has been preprocessed for you. Let us load them and take a look.


```python
import os
import pdb
import sys
import pickle
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# set random seed for reproducibility
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
os.environ['PYTHONASHSEED'] = str(seed)

# define data set path
DATA_PATH = '../HW5_MedGAN-lib/data'
```


```python
pids = pickle.load(open(os.path.join(DATA_PATH,'train/pids.pkl'), 'rb'))
vids = pickle.load(open(os.path.join(DATA_PATH,'train/vids.pkl'), 'rb'))
hfs = pickle.load(open(os.path.join(DATA_PATH,'train/hfs.pkl'), 'rb'))
seqs = pickle.load(open(os.path.join(DATA_PATH,'train/seqs.pkl'), 'rb'))
types = pickle.load(open(os.path.join(DATA_PATH,'train/types.pkl'), 'rb'))
rtypes = pickle.load(open(os.path.join(DATA_PATH,'train/rtypes.pkl'), 'rb'))

assert len(pids) == len(vids) == len(hfs) == len(seqs) == 1000
assert len(types) == 619
```

where

- `pids`: contains the patient ids
- `vids`: contains a list of visit ids for each patient
- `hfs`: contains the heart failure label (0: normal, 1: heart failure) for each patient
- `seqs`: contains a list of visit (in ICD9 codes) for each patient
- `types`: contains the map from ICD9 codes to ICD-9 labels
- `rtypes`: contains the map from ICD9 labels to ICD9 codes

Let us take a patient as an example.


```python
# take the 3rd patient as an example
print("Patient ID:", pids[3])
print("Heart Failure:", hfs[3])
print("# of visits:", len(vids[3]))
for visit in range(len(vids[3])):
    print(f"\t{visit}-th visit id:", vids[3][visit])
    print(f"\t{visit}-th visit diagnosis labels:", seqs[3][visit])
    print(f"\t{visit}-th visit diagnosis codes:", [rtypes[label] for label in seqs[3][visit]])
```


```python
print("number of heart failure patients:", sum(hfs))
print("ratio of heart failure patients: %.2f" % (sum(hfs) / len(hfs)))
```

## 1 Build Dataloader

### 1.1 CustomDataset

First of all, let's implement a custom dataset using PyTorch class `Dataset`, which will characterize the key features of the dataset we want to generate.

We will use the sequences of diagnosis codes `seqs` as input and heart failure `hfs` as output.


```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, seqs, hfs):
        '''
        TODO: Store `seqs`. to `self.x` and `hfs` to `self.y`.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        Do NOT permute the data.
        '''
        # your code here
        raise NotImplementedError
    
    def __len__(self):
        '''
        TODO: Return the number of samples (i.e. patients).
        '''
        # your code here
        raise NotImplementedError

    def __getitem__(self, index):
        '''
        TODO: Generates one sample of data.
        
        Note that you DO NOT need to covert them to tensor as we will do this later.
        '''
        return self.x[index], self.y[index]


dataset = CustomDataset(seqs, hfs)
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
dataset = CustomDataset(seqs, hfs)
assert len(dataset) == 1000


```

### 1.2 Data Collator

After building the dataset, we shall build a data collator.

This collate function `Collator()` will be called by `DataLoader` after fetching a list of samples using the indices from `CustomDataset` to collate the list of samples into batches.

For example, when the `DataLoader` gets a list of two samples.

```
[ [ [0, 1, 2], [3, 0] ], 
  [ [1, 3, 6, 3], [2], [3, 1] ] ]
```

where the first sample has two visits `[0, 1, 2]` and `[3, 0]` and the second sample has three visits `[1, 3, 6, 3]`, `[2]`, and `[3, 1]`.

The collate function `Collator()` is supposed to concatenate all visits of one patient together to form the inputs for MedGAN, as

```
[[0, 1, 2, 3 ,0],
 [1, 3, 6, 3, 2, 3 ,1]]

```

Further, we transform this to a multi-hot vector representing the appearances of events. Suppose we the number of all possible events is 6, the yielded outputs should be

```
[[0, 1, 1, 1, 0, 0],
 [0, 1, 1, 0, 0, 1]]
```
which will be the final inputs.




```python
class Collator:
    def __init__(self, total_number_of_codes):
        self.max_num_codes = total_number_of_codes
        
    def __call__(self, data):
        '''TODO: flatten the input sequence samples into a multi-hot diagnosis codes,
        e.g., a multi-hot codes [1, 0, 1] indicates the appearance of [code1, code3] in this patient's all visits.
        Arguments:
            data: a list of samples fetched from 'CustomDataset'

        Outputs:
            x: a tensor of shape (# patients, max # diagnosis codes) with torch.float
            y: a tensor of shape (# patients, ) with type torch.float
        '''
        sequences, labels = zip(*data)
        num_patients = len(sequences)
        max_num_codes =  self.max_num_codes
        y = torch.tensor(labels, dtype=torch.float)
        x = torch.zeros((num_patients, max_num_codes), dtype=torch.float)
        
        for i_patient, patient in enumerate(sequences):
            '''TODO: Update `x` by looping over each patient.
            '''
            # your code here
            raise NotImplementedError
        return x, y

collate_fn = Collator(len(types))
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

from torch.utils.data import DataLoader
dataset = CustomDataset(seqs, hfs)
collate_fn = Collator(len(types))
loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)
loader_iter = iter(loader)
x, y = next(loader_iter)

assert x.dtype == torch.float
assert y.dtype == torch.float
assert x.shape == (10, 619)
assert y.shape == (10,)


```

## 2 Naive AutoEncoder

Let's implement a naive AutoEncoder as done in original MedGAN. The first stage is to learn an AutoEncoder by taking reconstruction of the input $x$ by predicting $\hat{x}$.

<img src="img/medgan.png" width="400" />

As done in `Dataset` and `DataLoader`, the input $x$ is a binary vector with size (batch_size, # diag codes). 

We can make a simple encoder with one hidden linear layer `nn.Linear` by transforming $x$ to representations $h$ with size (batch size, hidden dimension). Then, we make a decoder also with one hidden linear layer `nn.Linear` by transforming $h$ to $\hat{x}$ with the same size as $x$. 

We will take `nn.Sigmoid` as the prediction activation to map logits $\hat{x}$ to $[0,1]$.

The detailed model architecture for you to follow is shown in the table below.

Layers | Configuration | Activation Function | Output Dimension (batch, feature)
--- | --- | --- | ---
fully connected | input size **input_dim**, output size **hidden_dim** | Tanh | (batch_size, hidden_dim)
fully connected | input size **hidden_dim**, output size **input_dim** | Sigmoid | (batch_size, input_dim)

### 2.1 Build the AutoEncoder model


```python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        '''TODO:
        initialize an auto-encoder
        self.encoder: linear - tanh activation
        self.decoder: linear - sigmoid activation
        
        Note: try to use nn.Sequential to stack layers and assign the block to self.encoder and self.decoder.
        '''
        super().__init__()
        
        # DO NOT change the names
        self.encoder = None
        self.decoder = None
        
        # your code here
        raise NotImplementedError
    
    def encode(self, x):
        '''TODO:
        take the input patient records, encode them into hidden representations
        using the encoder.
        Arguments:
            x: the patient records with shape (batch_size, max # diagnosis codes)
        Outputs:
            h: the encoded representations with shape (batch_size, hidden_dim)
        '''
        # your code here
        raise NotImplementedError
    
    def decode(self, h):
        '''TODO:
        take the input hidden representations, output the reconstructed patient records
        using the decoder.
        Arguments:
            h: the encoded representations with shape (batch_size, hidden_dim)
        Outputs:
            x: the patient records with shape (batch_size, max # diagnosis codes)
        '''
        # your code here
        raise NotImplementedError
        
    def forward(self, x):
        '''TODO:
        call the self.encode and self.decode and finally output the reconstructed input x.
        Arguments:
            x: the patient records with shape (batch_size, max # diagnosis codes)
        Outputs:
            x: the reconstructed patient records with shape (batch_size, max # diagnosis codes)
        '''
        # your code here
        raise NotImplementedError

model = AutoEncoder(1000, 256)
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
x = torch.randint(0, 2, (32, 1000)).float()
h = model.encode(x)
x_hat = model.decode(h)
assert isinstance(model.encoder, nn.Sequential), 'You should implement your encoder using nn.Sequential, found {}.'.format(type(model.encoder))
assert isinstance(model.decoder, nn.Sequential), 'You should implement your decoder using nn.Sequential, found {}.'.format(type(model.decoder))
assert h.shape == torch.Size([32, 256]), 'The encoder output shape is wrong, expect [32, 256], got {}'.format(h.shape)
assert x_hat.shape == torch.Size([32, 1000]), 'The decoder output shape is wrong, expect [32, {}], got {}'.format(1000,x_hat.shape)


```

### 2.2 Train the AutoEncoder model

With the built AE model at hand, it is easy to follow the common practice to train AE using reconstruction loss.

Let's make use of the completed `CustomDataset`, `Collator`, and `AutoEncoder` to achieve this!


```python
'''
TODO: Define the optimizer (Adam) with learning rate 1e-3.
Define the loss_fn (loss function, nn.BCELoss).
Do the training in each iteration by
- forward ae model to get x_hat
- compute reconstruction loss
- call loss.backward
- update parameters using optimizer.step
'''

from torch.utils.data import DataLoader
dataset = CustomDataset(seqs, hfs)
collate_fn = Collator(len(types))
dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
ae = AutoEncoder(input_dim=len(types), hidden_dim=256)

optimizer, loss_fn = None, None
# your code here
raise NotImplementedError

loss_list = []
for epoch in range(50):
    epoch_loss = 0
    for (x,y) in dataloader:
        optimizer.zero_grad()
        
        # your code here
        raise NotImplementedError
        
        epoch_loss += loss.item()
    loss_list.append(epoch_loss)
    print(f'epoch {epoch} training autoencoer loss {epoch_loss}')

import matplotlib.pyplot as plt
plt.plot(loss_list, label='autoencoder loss')
plt.xlabel('epoch')
plt.title('AE loss during training')
plt.legend()
plt.show()
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

```

## 3 MedGAN [100 points]

### 3.1 Build the Generator and Discrminator [30 points]

Next, we will construct the generator and discriminator of the MedGAN model.

Note that we take a conditional GAN here, the input of generator should be the concatenation of $z$ vector and the condition $y$. Meanwhile, the input of discriminator is the concatenation of $x$ vector and the condition $y$.

<img src="img/conditional-gan.png" width="400"/>

This figure is drawn from the paper [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf). In our case, the condition $y$ is a value in $\{0,1\}$ indicating whether the patient has heart failure or not.


The architecture details are as follows.

**Generator**

Layers | Configuration | Activation Function | Output Dimension (batch, feature)
--- | --- | --- | ---
batchnorm1d | - | - | (batch_size, input_dim)
ReLU | - | - |  (batch_size, input_dim)
fully connected | input size **input_dim**, output size **hidden_dim** | - | (batch_size, hidden_dim)
batchnorm1d | - | - | (batch_size, hidden_dim)
Tanh | - | - |  (batch_size, hidden_dim)
fully connected | input size **hidden_dim**, output size **hidden_dim** | - | (batch_size, hidden_dim)


**Discrminator**

Layers | Configuration | Activation Function | Output Dimension (batch, feature)
--- | --- | --- | ---
fully connected | input size **input_dim**, output size **hidden_dim** | - | (batch_size, hidden_dim)
ReLU | - | - |  (batch_size, input_dim)
fully connected | input size **hidden_dim**, output size **1** | - | (batch_size, 1)
Sigmoid | - | - |  (batch_size, 1)


```python
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        '''input records and labels for conditional generation
        
        TODO: define the layer components as shown in the above table.
        '''
        super().__init__()
        
        # DO NOT change the names
        self.linear1 = None
        self.bn1 = None
        self.act1 = None
        self.linear2 = None
        self.bn2 = None
        self.act2 = None
        
        # your code here
        raise NotImplementedError
        
    def forward(self, z, y):
        '''
        Arguments:
            z: input random noise with shape (n, hidden_dim)
            y: input conditional y with shape (n,)
        
        Outputs:
            h: the generated representation h with shape (n, hidden_dim)
        
        TODO: take the defined components to do forward inference.
        
        Note: do not forget to take the *residual connection* for each layer as described in MedGAN paper,
        i.e., (z,y) -> tmp -> layer1(bn1+act1+linear1) -> h -> h = h + z -> bn2 -> z -> layer2(act2+linear2) -> h -> h = h+z
        
        '''
        # concatenate the input z and condition y
        tmp = torch.cat([z, y[:,None]], 1)
        
        # your code here
        raise NotImplementedError
        
        return h
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        '''input records and labels for conditional discrimination
        
        TODO: define the layer components as shown in the above table.
        '''
        super().__init__()
        
        # DO NOT change the names
        self.linear1 = None
        self.act1 = None
        self.linear2 = None
        self.act2 = None
        
        # your code here
        raise NotImplementedError
        
    def forward(self, x, y):
        '''
        Arguments:
            x: input records with shape (n, input_dim)
            y: input conditional labels with shape (n,)
        Outputs:
            out: the predicted probability if input x is real or fake samples in shape (n,)
        
        Note: unlike in Generator, we DO NOT take residual connection here.
        '''
        
        # concatenate the input x and condition y
        x = torch.cat([x, y[:,None]], axis=1)
        out = None
        
        # your code here
        raise NotImplementedError
        
        return out.squeeze(1)

generator = Generator(100, 99)
discriminator = Discriminator(100, 10)
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
assert generator.linear1.in_features==100 and generator.linear1.out_features==99
assert generator.bn1.num_features==100
assert isinstance(generator.act1, nn.ReLU)
assert generator.linear2.in_features==99 and generator.linear1.out_features==99
assert generator.bn2.num_features==99
assert isinstance(generator.act2, nn.Tanh)


```

### 3.2 Build the MedGAN model [30 points]


Now, finally we come to build our conditional MedGAN with all completed components: AutoEncoder, Generator, Discriminator.

Recall the MedGAN architecture as

<img src="img/medgan.png" width="400" />

In the MedGAN training phase, only the `decoder` of the AutoEncoder model will be used to decode the outputs of the `generator`.

Also, we need a `generate` function for `MedGAN` such that it is able to generate fake samples for `discriminator` to classify.

In the `forward` function, we need to implement the compution for discriminator and generator loss.

$\ell_{d}= - \frac1m \sum_{i=1}^m [log (D(x_i)+\epsilon) + \log (1-D(\hat{x}_i)+\epsilon)]$

$\ell_{g}= - \frac1m \sum_{i=1}^m log (D(\hat{x}_i)+\epsilon)$

where $x_i$ is the real record and $\hat{x}_i$ is the fake record generated by generator;

$\epsilon$ is added to avoid numerical issue in the $log$ function.



```python
class MedGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        '''The main class for MedGAN model. It consists of three parts:
        AutoEncoder
        Generator
        Discriminator
        '''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        ### DO NOT change the names
        self.ae = AutoEncoder(input_dim, hidden_dim)
        self.generator = Generator(hidden_dim+1, hidden_dim) # input random noise/representation + label
        self.discriminator = Discriminator(input_dim+1, hidden_dim) # input records + label
            
    def generate(self, n, y):
        '''
        Arguments:
            n: number of fake samples to be generated
            y: the condition label used to make conditional generation
        Outputs:
            x: the generated fake samples with shape (n, self.input_dim)
        
        TODO: Generate n fake samples using the generator.
        
        First, sample a random vector z using torch.randn with size (n, self.hidden_dim).
        Then, generate the fake encoded representations h using z and y as inputs for self.generator .
        Last, generate the fake example x using self.ae.decode function.
        '''
        
        # your code here
        raise NotImplementedError
    
    def forward(self, x, y):
        '''Take the input x and conditional y, compute the discriminator loss and generator loss
        Arguments:
            x: input records with shape (n, self.input_dim)
            y: input labels with shape (n,)
        Outputs:
            d_loss: discriminator loss values
            g_loss: generator loss values
            
        TODO: Implement the prediction of fake or real examples using self.discriminator.
        Then, compute the discriminator loss and generator loss.
        '''
        # generate fake samples by putting random noise z into the generator
        x_fake = self.generate(len(x), y)
        
        fake_score = None
        real_score = None
        
        # discriminate x and x_fake using discriminator
        # your code here
        raise NotImplementedError
        
        # compute generator loss and discriminator loss, take epsilon for numerical stability
        d_loss = None
        g_loss = None
        epsilon = 1e-8
        
        # your code here
        raise NotImplementedError
        
        return g_loss, d_loss

medgan = MedGAN(100, 128)
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
x = torch.ones((2,100))
y = torch.ones((2))
out = medgan(x,y)
assert isinstance(medgan.ae, AutoEncoder), 'medgan.ae should be AutoEncoder!'
assert medgan.generator.linear1.in_features==129
assert medgan.generator.linear1.out_features==128
assert medgan.discriminator.linear1.in_features==101
assert medgan.discriminator.linear1.out_features==128
assert isinstance(medgan.generator, Generator), 'medgan.generator should be Generator!'
assert isinstance(medgan.discriminator, Discriminator), 'medgan.discriminator should be Discriminator!'



```

### 3.3 Build optimizers for MedGAN [20 points]

Now we turn to build optimizers for MedGAN. Note that GAN model trains generator and discriminator in an adversarial paradigm, we need to split their parameters when designing optimizers.



```python
def build_optimizer(medgan):
    '''build two separate optimizers for the generator and discriminator, respectively.
    
    TODO: add params which belong to AutoEncoder and Generator to g_param_list;
    add params which belong to discriminator to d_param_list.
    '''
    g_param_list, d_param_list = [], []
    for name, param in medgan.named_parameters():
        # your code here
        raise NotImplementedError
    g_optimizer = torch.optim.Adam(g_param_list, lr=1e-4)
    d_optimizer = torch.optim.Adam(d_param_list, lr=1e-4)
    return g_optimizer, d_optimizer
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

```

### 3.4 Train MedGAN and generate synthetic records [20 points]

Finally, we come to train the MedGAN we implement and use it to generate synthetic records we want.


```python
'''
TODO: complete the part where we use to update parameters of generator by g_loss and parameters of discriminator by d_loss.
'''

# build dataloader
from torch.utils.data import DataLoader
dataset = CustomDataset(seqs, hfs)
collate_fn = Collator(len(types))
dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

# build model
medgan = MedGAN(input_dim=len(types), hidden_dim=256)

# apply the pretrained ae model to medgan autoencoder model
medgan.ae.load_state_dict(ae.state_dict())

# build optimizer for generator and discriminator, respectively
g_opt, d_opt = build_optimizer(medgan)

n_epochs = 40
d_train_period = 1
g_train_period = 2

g_loss_list, d_loss_list = [], []
for epoch in range(n_epochs):
    medgan.train()
    g_loss_all, d_loss_all = 0, 0
    for (x,y) in dataloader:
        for _ in range(d_train_period):
            
            # your code here
            raise NotImplementedError
            
            d_loss_all += d_loss.item()

        for _ in range(g_train_period):
            # your code here
            raise NotImplementedError
    
    g_loss_list.append(g_loss_all)
    d_loss_list.append(d_loss_all)
    print(f'Epoch {epoch} Generator loss {g_loss_all} Discriminator loss {d_loss_all}')
```


```python
import matplotlib.pyplot as plt
plt.plot(g_loss_list, label='generator loss')
plt.plot(d_loss_list, label='discriminator loss')
plt.xlabel('epoch')
plt.title('GAN loss during training')
plt.legend()
plt.show()
```

To verify if our MedGAN is able to do conditional generation, we design an experiment like:

- Train a heart failure classifier on the real EHRs record from which MedGAN learns.
- Use the trained MedGAN to generate synthetic records with given condition in $\{0,1\}$
- Use the trained classifier to make predictions on synthetic records to see if the predicted outcomes match the given condition.

If the predicted outcome matches the given condition well, our MedGAN does a great job because it succeeds to produce records which are fit to the given condition. Let's start!


```python
# train a classifier on the real records so we can classify if one patient record belongs to heart failure case
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        pred = self.clf(x).squeeze(1)
        return pred

from torch.utils.data import DataLoader
dataset = CustomDataset(seqs, hfs)
collate_fn = Collator(len(types))
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
clf = Classifier(len(types), 256)
optimizer = torch.optim.Adam(clf.parameters(), 1e-3)
loss_fn = nn.BCELoss()
for epoch in range(10):
    epoch_loss = 0
    for (x,y) in dataloader:
        optimizer.zero_grad()
        pred = clf(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'epoch {epoch} training classifier loss {epoch_loss}')
```

Now let's generate synthetic records and test! 


```python
print('generate 100 synthetic records with/without heart failure')
y = torch.cat([torch.ones(50), torch.zeros(50)], 0)
medgan.eval()
with torch.no_grad():
    x = medgan.generate(100, y)
    pred = clf(x)


print('evaluate how much the generated synthetic records match the given condition:')    
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y.long().numpy(), pred.numpy())
print('auc:', auc)
    
pred[pred>0.5]=1
pred[pred<=0.5]=0
acc = (pred == y).float().mean().item()
print('accuracy:', acc)
```

Since training a GAN model has long been "random", it is possible that your model doesn't perform so well.
The reason might be 

(1) the training records are so small. Here we only have 1000 records.

(2) the conditional GAN does not capture the given condition well.

You can look into the references regarding conditional GAN later to think about how to improve conditional GAN for syntheti records generation.


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''

```


```python

```
