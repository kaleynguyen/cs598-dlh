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
### Customized unprocessed dataset
```
root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"
from pyhealth.datasets import BaseDataset
from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets.utils import strptime
from tqdm import tqdm

class CustomMIMIC(BaseDataset):
    def parse_tables(self):
        # patients is a dict of Patient objects indexed by patient_id
        patients = dict()
        # process patients and admissions tables
        patients = self.parse_basic_info(patients)
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                patients = getattr(self, f"parse_{table.lower()}")(patients)
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    """ THE VERY FIRST FUNCTION: clean patient-visit structure
        
        Note: You will create an empty nested dictionary structrue
          - level 1: patients: patient_id -> Patient
          - level 2: Patient.visit: visit_id -> Visit
    """
    def parse_basic_info(self, patients):

        # load patients info
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000, # set the first 100 rows for example
        )

        # load admission info
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )

        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")

        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)

        # load patients
        for p_id, p_info in tqdm(df.groupby("SUBJECT_ID"), desc="Parsing PATIENTS and ADMISSIONS"):
            # check <pyhealth.data.Patient> in https://pyhealth.readthedocs.io/en/latest/api/data/pyhealth.data.Patient.html
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                # check <pyhealth.data.Patient> in https://pyhealth.readthedocs.io/en/latest/api/data/pyhealth.data.Visit.html
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            # add patient
            patients[p_id] = patient
        return patients

      
    """ DIAGNOSES_ICD Process Function
        
        Note: You will insert the diagnosis info into each visit
            - the function name is '_parse_' + lowercase(csv_table_name) 
    """
    def parse_diagnoses_icd(self, patients):

        # the table name
        table = "DIAGNOSES_ICD"

        # load csv table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )

        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])

        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)

        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(df.groupby(["SUBJECT_ID", "HADM_ID"]), desc=f"Parsing {table}"):
            for code in v_info["ICD9_CODE"]:
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="ICD9CM",
                    visit_id=v_id,
                    patient_id=p_id,
                )
                # update patient-visit structure
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    """ PROCEDURES_ICD Process Function
        
        Note: You will insert the diagnosis info into each visit
            - the function name is '_parse_' + lowercase(csv_table_name) 
    """
    def parse_procedures_icd(self, patients):

        # the table name
        table = "PROCEDURES_ICD"

        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )

        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])

        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)

        # iterate over each patient and visit
        for (p_id, v_id), v_info in tqdm(df.groupby(["SUBJECT_ID", "HADM_ID"]), desc=f"Parsing {table}"):
            for code in v_info["ICD9_CODE"]:
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="ICD9PROC",
                    visit_id=v_id,
                    patient_id=p_id,
                )
                # update patient-visit structure
                patients = self._add_event_to_patient_dict(patients, event)
        return patients
dataset = CustomMIMIC(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"],
)

```
### Sample Data set / Balanced simulated data
```
import numpy as np

# simulate N_pat patients
# simulate N_vis visit per patient
# simulate the conditions per visit
# simulate the lab_values per visit
# simualte the binary labels per visit
N_pat, N_vis = 50, 10
condition_space = [f"cond-{i}" for i in range(100)]

samples = []
for pat_i in range(N_pat):
    conditions = []
    lab_values = []
    for visit_j in range(N_vis):
        patient_id = f"patient-{pat_i}"
        visit_id = f"visit-{visit_j}"
        # how many conditions to simulate
        N_cond = np.random.randint(3, 6)
        conditions.append(np.random.choice(condition_space, N_cond, replace=False).tolist())
        # how many lab-values to simulate
        lab_values.append(np.random.random(5).tolist())
        # which binary label
        label = int(np.random.random() > 0.5)
        
        # make sure to store historical visits information into the current visit as well.
        sample = {
            "patient_id": patient_id,
            "visit_id": visit_id,
            "conditions": conditions.copy(),
            "lab-values": lab_values.copy(),
            "label": label
        }
        samples.append(sample)


```
# Step 2. Tasks
### Default task
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
### customized tasks
```
def mortality_prediction(patient):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []

    # loop over all visits but the last one
    for i in range(len(patient) - 1):

        # visit and next_visit are both <pyhealth.data.Visit> objects
        visit = patient[i]
        next_visit = patient[i + 1]

        # step 1: define the mortality_label
        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # step 3: exclusion criteria: visits without condition, procedure, or drug
        if len(conditions) * len(procedures) * len(drugs) == 0: continue
        
        # step 4: assemble the samples
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": mortality_label,
            }
        )
    return samples
mor_dataset = dataset.set_task(mortality_prediction)
mor_dataset.stat()
```


# Step 3. Models
### Default model such as transformer or RETAIN
```
from pyhealth.models import Transformer
model = Transformer(
    dataset=mimic3_ds,
    feature_keys=['conditions', 'procedures'],
    label_key='label',
    mode='binary'
)
```
### Customized model
```
from typing import Tuple, List, Dict, Optional
import functools

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
class MyPyTorchDeepr(nn.Module):
    def __init__(self, feature_size: int = 100, 
                 window: int = 1, 
                 hidden_size: int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(feature_size, hidden_size, kernel_size=2 * window + 1)

    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None: x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # [batch size, input size, sequence len]
        x = torch.relu(self.conv(x))
        return x.max(-1)[0]

class MyPyHealthDeepr(BaseModel):
    def __init__(self, dataset: BaseDataset, feature_keys: List[str], label_key: str, mode: str, 
                 embedding_dim=128, hidden_dim=32, **kwargs):
        super().__init__(dataset, feature_keys, label_key, mode)

        # Any BaseModel should have these attributes, as functions like add_feature_transform_layer uses them
        self.feat_tokenizers = {}
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embedding_dim = embedding_dim

        # self.add_feature_transform_layer will create a transformation layer for each feature
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            self.add_feature_transform_layer(
                feature_key, input_info, special_tokens=["<pad>", "<unk>", "<gap>"]
            )

        # We create one MyPyTorchDeepr for each feature (e.g. drug, diagnosis, etc.)
        self.deepr_modules = nn.ModuleDict()
        for feature_key in feature_keys:
            self.deepr_modules[feature_key] = MyPyTorchDeepr(feature_size=embedding_dim, hidden_size=hidden_dim, **kwargs)
        
        # final output layer 
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]

            # each patient's feature is represented by [[code1, code2],[code3]]
            assert input_info["dim"] == 3 and input_info["type"] == str

            # Aggregating features according to deepr's requirement 
            feature_vals = [
                    functools.reduce(lambda a, b: a + ["<gap>"] + b, _)
                    for _ in kwargs[feature_key]
                ]
            
            x = self.feat_tokenizers[feature_key].batch_encode_2d(feature_vals, padding=True, truncation=False)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            pad_idx = self.feat_tokenizers[feature_key].vocabulary("<pad>")
            #create the mask
            mask = (x != pad_idx).long()
            embeds = self.embeddings[feature_key](x)
            feature_embed = self.deepr_modules[feature_key](embeds, mask)
            patient_emb.append(feature_embed)

        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}
# Stage 3: define model
device = "cpu"

model = MyPyHealthDeepr(
    dataset=dataset,
    feature_keys=["conditions", "drugs", "procedures"],
    label_key="label",
    mode="binary",
    embedding_dim=128,
    hidden_dim=64,
)
model.to(device)

# Stage 4: model training
from pyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=3,
    monitor="pr_auc",
)
# evaluation option 1: use our built-in evaluation metric
result = trainer.evaluate(test_loader)
print ('\n', result)

# evaluation option 2: use pyhealth.metrics
from pyhealth.metrics.binary import binary_metrics_fn
y_true, y_prob, loss = trainer.inference(test_loader)
result = binary_metrics_fn(y_true, y_prob, metrics=["pr_auc", "roc_auc"])
print ('\n', result)

# evaluation option 3: use sklearn.metrics
from sklearn.metrics import average_precision_score, roc_auc_score
y_pred = (y_prob > 0.5).astype('int')
print (
    '\n',
    'roc_auc', roc_auc_score(y_true, y_prob), 
    'pr_auc:', average_precision_score(y_true, y_prob)
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
