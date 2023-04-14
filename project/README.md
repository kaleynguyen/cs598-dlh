# Pyhealth General Objects: 

Each patient goes to different hospitals. At each hospital, they have multiple visits. At each visit, they have multiple events such as diagnosis event, procedure event, medication event or lab order / lab test event. 

### pyhealth.data.Event

```
from pyhealth.data import Event
from datetime import datetime
event1 = Event(
    code="428.0",
    table="DIAGNOSES_ICD",
    vocabulary="ICD(CM",
    visit_id="v001",
    patient_id="p001",
    timestamp=datetime.fromisoformat("2019-08-12 00:00:00"),
    active_on_discharge=True #additional att, can access by event1.attr_dict
)
```

### pyhealth.data.Visit

```
from pyhealth.data import Visit
from datetime import datetime, timedelta

# create a visit
visit1 = Visit(
    visit_id="v001",
    patient_id="p001",
    encounter_time=datetime.now() - timedelta(days=2),
    discharge_time=datetime.now() - timedelta(days=1),
    discharge_status='Alive',
)
visit1.add_event(event1)
visit1.available_tables
visit1.num_events
visit1.event_list('DIAGNOSES_ICD')
visit1.code_list('DIAGNOSES_ICD')

``` 

### pyhealth.data.Patient

```
from pyhealth.data import Patient
from datetime import datetime, timedelta

# patient is a <Patient> instance with many attributes

patient = Patient(
    patient_id="p001",
    birth_datetime=datetime(2012, 9, 16, 0, 0),
    death_datetime=None,
    gender="F",
    ethnicity="White",
)

# add visit
patient.add_visit(visit1)
patient.add_visit(visit2)
# add event
patient.add_event(event1)
# other methods
patient.get_visit_by_id("v001")
patient.get_visit_by_index(0)

print(patient)
```


# Whole pipeline
# Step 1. Data / Datasets
### Mimic3 dataset
```
from pyhealth.datasets import MIMIC3Dataset

mimic3_ds = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True
)
mimic3_ds.stat()
mimic3_ds.info()
```
### Customized dataset
```
root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"

```
# Step 2. Tasks
```
from pyhealth.tasks import mortality_prediction_mimic3_fn
mimic3_ds = mimic3_ds.set_task(task_fn=mortality_prediction_mimic3_fn)
mimic3_ds.stat()
```
```
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(mimic3_ds, [0.8, 0.1, 0.1])

# create dataloaders (they are <torch.data.DataLoader> object)
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)
```
# Step 3. Models
```
from pyhealth.models import Transformer
model = Transformer(
    dataset=mimic3_ds,
    feature_keys=['conditions', 'procedures'],
    label_key='label',
    mode='binary'
)
```
# Step 4. Trainer
```
from pyhealth.trainer import Trainer
trainer = Trainer(model=model)
trainer.train(
train_dataloader=train_loader,
val_dataloader=val_loader,
epochs=3,
monitor='pr_auc')
```
# Step 5. Metrics / Eval
```
from pyhealth.metrics.binary import binary_metrics_fn
y_true, y_prob, loss = trainer.inference(test_loader)
binary_metrics_fn(y_true, y_prob, metrics=["pr_auc", "roc_auc", "f1"])
```
# Step 6. Calibration and Uncertainty Quantification
