# **HW2_pyhealth_medcode**

## Overview

```pyhealth.medcode``` is the medical code tool in pyhealth, which provides two core functionalities: 

- (1) looking up information for a given medical code (e.g., name, category, sub-concept); 
- (2) mapping codes across coding systems (e.g., ICD9CM to CCSCM). 

The documentation is at https://pyhealth.readthedocs.io/en/latest/api/medcode.html.
    
This assignment is designed to get you familiar with ```pyhealth.medcode```. After finishing the homework, you may use the module to help implement part of the final projects. 

## 1 Code Lookup

`class pyhealth.medcode.InnerMap`

Documentation at https://pyhealth.readthedocs.io/en/latest/api/medcode.html#pyhealth.medcode.InnerMap

### Functionality
- lookup(code): looks up code in a coding system
- contains(code): checks whether a code belongs to a specific coding system
- get_ancestors(code): returns the ancestors for a given code

Currently, we support the following coding systems:

- Diagnosis codes:
    - ICD9CM
    - ICD10CM
    - CCSCM
- Procedure codes:
    - ICD9PROC
    - ICD10PROC
    - CCSPROC
- Medication codes:
    - NDC
    - RxNorm
    - ATC

## 1.1 Look up ICD9CM code

Let's first try to look up the ICD9 CM code 428.0, which stands for "Congestive heart failure, unspecified".


```python
from pyhealth.medcode import InnerMap
icd9cm = InnerMap.load("ICD9CM")

# let's first check if the code is in ICD9CM
"428.0" in icd9cm
```


```python
# next let's look up this code
icd9cm.lookup("428.0")
```


```python
# we can also get the ancestors of this code
icd9cm.get_ancestors("428.0")
```

Note that if the code is not in standard format (e.g., "4280" instead of "428.0"), medcode will automatically normalize it.


```python
# non-standard format
icd9cm.lookup("4280")
```

### TODO: look up the following ICD9CM codes: 480.1, 280, 394


```python
icd9_4801 = icd9cm.lookup('4801')
icd9_280 = icd9cm.lookup('280')
icd9_394 = icd9cm.lookup('394')

# your code here
#raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

### TODO: look up the following CCSPROC codes: 2, 5, 10


```python
ccsproc = InnerMap.load("CCSPROC")

ccs_2 = ccsproc.lookup('2')
ccs_5 = ccsproc.lookup('5')
ccs_10 = ccsproc.lookup('10')

# your code here
#raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 1.2 Look up ATC code

For the medication code ATC, medcode provides additional information from DrugBank,.


```python
atc = InnerMap.load("ATC")

# let's search for M01AE51
atc.lookup("M01AE51")
```


```python
#  DrugBank ID
print(atc.lookup("M01AE51", "drugbank_id"))
```


```python
#  Drug description from DrugBank
print(atc.lookup("M01AE51", "description"))
```


```python
#  Drug indication from DrugBank
print(atc.lookup("M01AE51", "indication"))
```


```python
#  Drug SMILES string from DrugBank
print(atc.lookup("M01AE51", "smiles"))
```

### TODO: look up the drugbank_id, descriptions, indications and smiles for ATC code: B01AC06 (Aspirin)


```python
atc_dbid_B01AC06 = atc.lookup("B01AC06", "drugbank_id")
atc_description_B01AC06 = atc.lookup("B01AC06", "description")
atc_indication_B01AC06 = atc.lookup("B01AC06", "indication")
atc_smiles_B01AC06 = atc.lookup("B01AC06", "smiles")

# your code here
#raise NotImplementedError
```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```

## 2. Code Mapping

`class pyhealth.medcode.CodeMap`

Documentation at https://pyhealth.readthedocs.io/en/latest/api/medcode.html#pyhealth.medcode.CrossMap

### Args
- source: source code vocabulary to map from
- target: target code vocabulary to map to

### Functionality
- map(source_code): maps source_code to the target vocabulary

Currently, we support the following mapping:

- With in diagnosis codes:
    - ICD9CM <-> CCSCM
    - ICD10CM <-> CCSCM
- With in procedure codes:
    - ICD9PROC <-> CCSPROC
    - ICD10PROC <-> CCSPROC
- With in medication codes:
    - NDC <-> RxNorm
    - NDC <-> ATC
    - RxNorm <-> ATC
- Between diagnosis and medication codes:
    - ATC <-> ICD9CM

## 2.1 Map ICD9CM code to CCSCM code

Let's try to map the ICD9 CM code 428.0, which stands for "Congestive heart failure, unspecified", to CCS CM code.


```python
from pyhealth.medcode import CrossMap

mapping = CrossMap.load(source_vocabulary="ICD9CM", target_vocabulary="CCSCM")
mapping.map("428.0")
```

Note that the returned variable is a list of codes, due to the possible one-to-many mapping.


```python
# let's check if the mapping is correct
ccscm = InnerMap.load("CCSCM")
ccscm.lookup("108")
```

## 2.2 Map NDC code to ATC code

Let's try to map the NDC code 5058060001, which is acetaminophen 325 MG Oral Tablet [Tylenol].

See https://fda.report/NDC/50580-496.


```python
from pyhealth.medcode import CrossMap

mapping = CrossMap.load("NDC", "RxNorm")
mapping.map("50580049698")
# (please be patient, loading NDC and RxNorm tables...
# it may take up to 5 minutes.)
```


```python
mapping = CrossMap.load("RxNorm", "NDC")
# (please be patient, it may take up to 5 minutes)
mapping.map("209387")
```


```python
# let's check if the mapping is correct
ccscm = InnerMap.load("RxNorm")
ccscm.lookup("209387")
```

### TODO: Map NDC code 50090539100 to ATC.

See https://ndclist.com/ndc/50090-5391/package/50090-5391-0.


```python
mapping = CrossMap.load('NDC', 'ATC')
result = mapping.map('50090539100')

# your code here

```


```python
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''


```


```python

```
