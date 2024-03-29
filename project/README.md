# 1. Dagster pipeline

Run `dagster dev` command after installing dagster and dagit through pip. Select the right config file and run it with dagit UI to materialize the assets based on the steps below:

* Raw-files group stages the raw parquet tables into polars DataFrame and materialized the tables onto the disks as a set of software define assets.  
* Large-joined-files group combines 3 large Lazy DataFrames together, the admission table, the icustays table, and the patients table.
*  Extract-subjects group remove the transfer ICU stays, patients who have more than 2+ stays and pediatric patients. The joined table includes 61, 532 unique icu stays. After the filtering process, the total number of stays is 42276 as described in the MIMIC-III benchmark \cite{mimiciiibenchmark}. The mortality rate in hospital within 48 hours after admissions is $\frac{4471}{4471+30234}$ = $12.88\% $. The filtered icu stays and subjects are being used as the dependency asset of other assets within the same group. 
* Tokenize-itemid-by-episode group separates the continuous variables and discrete variables, discretizes the continuous variables by binning them onto 10-percentile per each itemid and append the string result of each itemid if it is discrete. The 10-percentiles are calculated based on a window function over each item id and then compare it with the subject\_id result of that itemid's value. 
* Subject-to-episode-allevents group creates a final array for training. There are 40119 total tokens. The number of total patients are 33790 and there are 3 columns: "subject\_id", "mortality\_tf" and "itemidx". The itemidx is the index of the item-to-index map that maps 40119 unique tokens to their associated indices. 

# 2. Notebooks
After obtaining the allevents_by_episode asset, run any of the notebooks and re-produce the result on a GPU-powered VM.  
