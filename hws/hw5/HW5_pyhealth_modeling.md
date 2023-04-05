# **HW5_pyhealth_modeling**

## Overview

In pyhealth, we treat each machine learning pipeline as five stages:
> **dataset process** -> **set healthcare task** -> **build ML model** -> **training** -> **inference/test**

They correspond to `pyhealth.datasets`, `pyhealth.tasks`, `pyhealth.models`, `pyhealth.trainer`, and `pyhealth.metrics`. We have alrady learned how to use the first two. Today, we will learn how to use the last three.

This assignment will go over a complete five-stage ML pipeline and build a clinical drug recommendation flow. After learning, you will be able to use pyhealth to build your own machine learning pipeline. You may build the final project on top of this five-stage framework.


```python
import warnings
warnings.filterwarnings('ignore')
```

### 1 Dataset Process
- Recall that we have learned how to process the open EHR data in the last HW following the [document](https://pyhealth.readthedocs.io/en/latest/api/datasets.html)
- In this assginment, we will
    - use the synthetic MIMIC-III dataset at https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/. 
        - You could replace the root with the real MIMIC-III dataset in your local environemnt and try it out.
    - process the three EHR tables: `DIAGNOSES_ICD.csv`, `PROCEDURES_ICD.csv`, `PRESCRIPTIONS.csv`.
    - apply the code mapping by transform the original NDC codes in PRESCRIPTIONS.csv to ATC-3 level code.
- **[Next Step]:** The output object will be used in **Step 2**.


```python
from pyhealth.datasets import MIMIC3Dataset

mimic3_ds = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
)

# we show the statistics below.
mimic3_ds.stat()
```

### 2 Set Drug Recommendation Task
- The last HW also taught us how to choose or define a new **healthcare task function** and use `.set_task()` to obtain samples. The **task function** specifics how to process each pateint's data into a set of samples for the downstream machine learning models. 
- Here, we use the default drug recommendation on MIMIC-III dataset from `pyhealth.tasks`. Users can also define their own function as well.
    - [Tutorials](https://colab.research.google.com/drive/1r7MYQR_5yCJGpK_9I9-A10HmpupZuIN-?usp=sharing) on how to define your own task function.
- **[Next Step]:** The outputs are a list of machine learning ready samples, which will be used in **Step 3**.




```python
from pyhealth.tasks import drug_recommendation_mimic3_fn

dataset = mimic3_ds.set_task(task_fn=drug_recommendation_mimic3_fn)
```

### TODO: obtain the dataset information

- Hint: use dataset.samples


```python
"""
TODO: please obtain the first sample
"""
first_sample = None
# your code here
raise NotImplementedError


"""
TODO: please count the number of total samples
""" 
num_of_samples = None
# your code here
raise NotImplementedError


"""
TODO: please count the number of patients in dataset
"""
num_of_patients = None
# your code here
raise NotImplementedError


"""
TODO: please count the number of visits in dataset
"""
num_of_visits = None
# your code here
raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 2.1 dataset split into pytorch data loader

The following code is used for dataset splitting into train/val/test sets.
- Here, we use the `split_by_patient` function to make sure each patient only goes into one of the sets. [[pyhealth.datasets.splitter]](https://pyhealth.readthedocs.io/en/latest/api/datasets/pyhealth.datasets.splitter.html) also provides other data split functions. We split the dataset into 80%:10%:10% and then load them into the standard pytorch data loader format.




```python
import numpy as np
np.random.seed(1234)

from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])

# create dataloaders (they are <torch.data.DataLoader> object)
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)
```

## 3. Build a ML Model

[pyhealth.models](https://pyhealth.readthedocs.io/en/latest/api/models.html) provides common deep learning models (e.g., RNN, CNN, Transformer) and special healthcare deep learning models, such as (e.g., RETAIN, SafeDrug, GAMENet). All except some special models (e.g., GAMENet, SafeDrug, MICRON are designed only for drug recommendation task) can be applied to all healthcare prediction tasks. 

- **[Arguments for Model]**:
  The arguments for each DL Model follows the arguments below.
    - `dataset`: this is the [pyhealth.datasets.SampleDataset](https://pyhealth.readthedocs.io/en/latest/api/datasets.html) object (output from step 2).
    - `feature_keys`: a list of string-based table names, indicating that these tables will be used in the pipeline.
    - `label_key`: currently, we only support `label`, defined in task function.
    - `mode`: `multiclass`, `multilabel`, or `binary`.
    - other specific model arguments, such as dropout, num_layers...
    
- In this assignment, we are going to 
    - utilize the "Transformer" model
    - choose "conditions", "procedures", and "drugs_all" as three features
    - choose "drugs" as the predicted targets (and it is a multilabel target).


### TODO: Let us first look at the form of one sample and then define a Transformer model


```python
# we print the 6-th sample
print (dataset.samples[5])
```


```python
"""
TODO: 
    - initialize a Transformer model by pyhealth.models.Transformer
    - use "conditions", "procedures", and "drugs_all" as the features
    - use "drugs" as the predicted targets
    - refer to https://colab.research.google.com/drive/1LcXZlu7ZUuqepf269X3FhXuhHeRvaJX5?usp=sharing
    
Hint:
    Transformer(
        dataset = ...,
        feature_keys = ...,
        label_key = ...,
        mode = ...,
    )
"""

from pyhealth.models import Transformer

model = None
# your code here
raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 4 Model Training

[pyhealth.trainer.Trainer](https://pyhealth.readthedocs.io/en/latest/api/trainer.html) is the training handler (similar to [pytorch-lightning.Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)). pyhealth uses it to train the ML and DL model. It has the following arguments and functionality

- **[Arguments]**: 
To initialize a trainer instance, the following environments should be specified.
  - `model`: the pyhealth.models object
  - `checkpoint_path (optional)`: path to intermediate checkpoint
  - `metrics (optional)`: which metrics to record during training. For example, we can record the pr_auc and auc_roc metrics. 
  - `device (optional)`: device to use
  - `enable_logging (optional)`: enable logging
  - `output_path (optional)`: output path
  - `exp_name (optional)`: experiment/task name

- **[Functionality]**:
  - `Trainer.train()`: simply call the `.train()` function will start to train the DL or ML model.
    - `train_dataloader`: train data loader
    - `val_dataloader`: validation data loader
    - `epochs`: number of epochs to train the model
    - `optimizer_class (optional)`: optimizer, such as `torch.optim.Adam`
    - `optimizer_params (optional)`: optimizer parameters, including
      - `lr (optional)`: learning rate
      - `weight_decay (optional)`: weight decay
    - `max_grad_norm (optional)`: max gradient norm
    - `monitor `: metric name to monitor, default is None
    - `monitor_criterion (optional)`: criterion to monitor, default is "max"
    - `load_best_moel_at_last (optional)`: whether to load the best model during the last iteration.

### TODO: initialize a trainer and train the model


```python
"""
TODO:
    - use Trainer() to initialize a trainer
        - record "jaccard_weighted" and "hamming_loss" during the training
        - use "cpu" as the device,
        - set the experiment name as "drug_recommendation"
        - hint:
        
            Trainer(
                model = ...,
                metrics = ...,
                device = ...,
                exp_name = ...,
            )
            
    - use trainer.train() to start the training
        - set epoch number to 20
        - monitor the "jaccard_weighted" metric, and use "max" as the criterion
        - load the best model when finishing training
        - hint:
            
            Trainer.train(
                train_dataloader = ...,
                val_dataloader = ...,
                epochs = ...,
                monitor = ...,
                monitor_criterion = ...,
            )
"""
from pyhealth.trainer import Trainer

trainer = None

# your code here
raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 5 Model Evaluation

[pyhealth.metrics](https://pyhealth.readthedocs.io/en/latest/api/metrics.html) contains the metrics for evaluating
  - [multiclass classification](https://pyhealth.readthedocs.io/en/latest/api/metrics/pyhealth.metrics.multiclass.html)
  - [multilabel classification](https://pyhealth.readthedocs.io/en/latest/api/metrics/pyhealth.metrics.multilabel.html)
  - [binary classification](https://pyhealth.readthedocs.io/en/latest/api/metrics/pyhealth.metrics.binary.html)
  
In this assginment, we use the multiclass classfication metrics.

## 5.1 one-line evaluation

The trainer has the `.evaluate(test_loader)` method to obtain the result metrics for any `test_loader` (the one you obtained in **2.1**).

### TODO: use trainer `.evaluate()` method to evaluate the test performance


```python
result = None
# your code here
raise NotImplementedError

print (result)
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 5.2 use pyhealth.metrics to evaluate

trainer has another method `.inference(test_loader)` to obtain the y_true, y_prob, loss for any test_loader (the one you obtained in 2.1). We will use it here and together with pyhealth.metrics to evaluate the performance.


```python
# obtain the true label, predicted probability, evaluation loss 
y_true, y_prob, loss = trainer.inference(test_loader)
```

### TODO: use pyhealth.metrics to obtain the following metrics on test data

- pr_auc_samples
- f1_weighted
- recall_macro
- precision_micro

check the [example](https://pyhealth.readthedocs.io/en/latest/api/metrics/pyhealth.metrics.multilabel.html)


```python
from pyhealth.metrics import multilabel_metrics_fn

result = None
# your code here
raise NotImplementedError

print (result)
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```


```python

```
