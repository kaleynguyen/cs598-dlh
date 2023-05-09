import polars as pl
pl.enable_string_cache(True)

from dagster import (asset, 
	op, 
	job, 
	graph, 
	get_dagster_logger, 
	Output, 
	MetadataValue, 
	Config, 
	OpExecutionContext,
	Definitions)
from datetime import date, timedelta
import numpy as np
from pydantic import Field

class MyAssetConfig(Config):
	data_path: str = Field(description="data path of the project. Check ./config.yaml")
	time_h: int = Field(description="time to truncate after the icustay")
	#seed: str = Field(description="random seed to split the data")

"""
Input: read patients data from the config path
Output: the patient DataFrame and its metadata 
"""
import os
@asset(group_name="raw_files")
def patients(config: MyAssetConfig) -> Output:
	logger = get_dagster_logger()
	logger.debug((os.getcwd()))
	df = pl.read_parquet(f'{config.data_path}/patients.parquet')
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}',
            "schema": f'{df.schema}',
        }
    )


@asset(group_name="raw_files")
def admissions(config: MyAssetConfig) -> Output:
	admissions = pl.read_parquet(f'{config.data_path}/admissions.parquet')
	df = admissions.drop('row_id')
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}',
            "schema": f'{df.schema}',
        }
    )


@asset(group_name="raw_files")
def icustays(config: MyAssetConfig) -> Output:
	icustays = pl.read_parquet(f'{config.data_path}/icustays.parquet')
	df = icustays.drop('row_id')
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}',
            "schema": f'{df.schema}',
        }
    )


@asset(group_name="large_joined_files")
def icu_adm_pts(icustays, admissions, patients):
	icustays = icustays.lazy().join(admissions.lazy(), left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])
	df = icustays.lazy().join(patients.lazy(), left_on=['subject_id'], right_on=['subject_id'])
	df = df.collect()
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}',
            "schema": f'{df.schema}',
        }
    )

@asset(group_name="extract_subjects")
def filtered_icu(icu_adm_pts):
	df = icu_adm_pts

	# Clean df
	logger = get_dagster_logger()
	logger.info(f'Before cleaning: {df.shape}')

	# Remove icu transfer
	df = icu_adm_pts.filter((pl.col('first_wardid') == pl.col('last_wardid')) & \
							(pl.col('first_careunit') == pl.col('last_careunit')))
	logger.info(f'After removing transfers: {df.shape}')

	# Remove 2+ stays per admissions
	df = df.with_columns(pl.col("icustay_id").count().over("hadm_id").suffix("_cnt"))\
			.filter(pl.col("icustay_id_cnt") == 1)
	logger.info(f'After removing 2+ stays: {df.shape}')	

	# Remove peds patients
	df = df.with_columns(((pl.col("intime") - pl.col("dob")) // timedelta(days=365.2425)).alias('age'))\
			.with_columns(pl.when(pl.col("age") < 0).then(pl.lit(90)).otherwise(pl.col("age")).alias("age"))\
			.filter((pl.col("age") >= 18) & (pl.col("age") < np.inf)) 
	logger.info(f'After removing peds patients: {df.shape}')		

	# Add mortality info during the filtered stays
	df = df.with_columns((pl.col("dod").is_not_null() | pl.col('deathtime').is_not_null()).alias('mortality'))
	df = df.with_columns(((pl.col("intime") <= pl.col("dod")) & (pl.col("outtime") >= pl.col("dod"))).alias('mortality_in_stay'))
	df = df.with_columns(((pl.col("admittime") <= pl.col("deathtime")) &  (pl.col("dischtime") >= pl.col("deathtime"))).alias('mortality_in_admission'))
	all_mortality = df.select(pl.col('mortality')).to_series().cast(pl.Boolean) &\
				(df.select(pl.col('mortality_in_stay')).to_series().cast(pl.Boolean) &\
				df.select(pl.col('mortality_in_admission')).to_series().cast(pl.Boolean))
	df = df.with_columns(all_mortality.alias('all_mortality').fill_null(0))	
	df = df.with_columns(pl.when(pl.col("mortality") != 0).then(pl.lit(True)).otherwise(False).alias('mortality_tf'))
	logger.info(f'Add mortality_tf: {df.head()}')
	groupby_mortality = df.groupby(['subject_id','mortality_tf'], maintain_order=True).agg(pl.count())
	groupby_mortality = groupby_mortality .groupby('mortality_tf').agg(pl.count())
	logger.info(f'Check mortality_tf count: {groupby_mortality}')
	
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}',
            "schema": f'{df.schema}',
        }
    )


@asset(group_name="extract_subjects")
def filtered_patients(config: MyAssetConfig, filtered_icu):
	patients = pl.read_parquet(f'{config.data_path}/patients.parquet')
	patients = patients.drop('row_id')
	df = patients.lazy().join(filtered_icu.lazy(), on=["subject_id"])
	df = df.groupby('subject_id').agg(pl.col("mortality_tf").any())
	df = df.collect()
	logger = get_dagster_logger()
	logger.info(df.head())
	logger.info(df.columns)
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}',
            "schema": f'{df.schema}',
        }
    )



@asset(group_name="extract_subjects")
def filtered_chartevents(config: MyAssetConfig, filtered_icu):
	chartevents = pl.read_parquet(f'{config.data_path}/chartevents.parquet')
	df = chartevents
	df = df.with_columns(pl.col("charttime").min().over("subject_id").suffix("_min"))
	df = df.with_columns(((pl.col('charttime')-pl.col('charttime_min')) // timedelta(minutes=60)).alias('hour'))\
		   .filter( (pl.col('hour') <= config.time_h) & (pl.col('hour') >= 0))
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}', 
            "schema": f'{df.schema}',
        }
    )

@asset(group_name="extract_subjects")
def filtered_labevents(config: MyAssetConfig, filtered_icu):
	labevents = pl.read_parquet(f'{config.data_path}/labevents.parquet')
	labevents = labevents.drop('row_id')
	subject_id = filtered_icu.select(pl.col('subject_id')).to_series()
	hadm_id = filtered_icu.select(pl.col('hadm_id')).to_series()
	df = labevents.lazy().filter(pl.col('subject_id').is_in(subject_id))\
					.filter(pl.col('hadm_id').is_in(hadm_id))
	df = filtered_icu.lazy().join(df, on=['subject_id', 'hadm_id'])
	df = df.filter( (pl.col('charttime') >= pl.col('intime')) & (pl.col('charttime') <= pl.col('outtime')))\
			.with_columns(((pl.col('charttime')-pl.col('intime')) // timedelta(minutes=60)).alias('hour'))\
			.filter( (pl.col('hour') <= config.time_h) & (pl.col('hour') >= 0))
	df = df.collect()
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}', 
            "schema": f'{df.schema}',
        }
    )
@asset(group_name="extract_subjects")
def filtered_outputevents(config: MyAssetConfig, filtered_icu):
	outputevents = pl.read_parquet(f'{config.data_path}/outputevents.parquet')
	outputevents = outputevents.drop('row_id')
	subject_id = filtered_icu.select(pl.col('subject_id')).to_series()
	hadm_id = filtered_icu.select(pl.col('hadm_id')).to_series()
	icustay_id = filtered_icu.select(pl.col('icustay_id')).to_series()
	df = outputevents.lazy().filter(pl.col('subject_id').is_in(subject_id))\
					.filter(pl.col('hadm_id').is_in(hadm_id))\
					.filter(pl.col('icustay_id').is_in(icustay_id))
	df = filtered_icu.lazy().join(df, on=['subject_id', 'hadm_id'])
	df = df.filter( (pl.col('charttime') >= pl.col('intime')) & (pl.col('charttime') <= pl.col('outtime')))\
			.with_columns(((pl.col('charttime')-pl.col('intime')) // timedelta(minutes=60)).alias('hour'))\
			.filter( (pl.col('hour') <= config.time_h) & (pl.col('hour') >= 0))
	df = df.collect()
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}', 
            "schema": f'{df.schema}',
        }
)

@asset(group_name="tokenize_itemid_by_episode")
def tokenize_all_events(filtered_chartevents, filtered_labevents, filtered_outputevents, filtered_patients):
	filtered_labevents = filtered_labevents.with_columns(pl.col('hour').cast(pl.Float64).alias('hour_float'))	
	filtered_chartevents = filtered_chartevents.with_columns(pl.col('hour').cast(pl.Float64).alias('hour_float'))	
	filtered_chartevents = filtered_chartevents.lazy().join(filtered_patients.lazy(), on=['subject_id'])
	filtered_chartevents = filtered_chartevents.collect()
	filtered_outputevents = filtered_outputevents.with_columns(pl.col('hour').cast(pl.Float64).alias('hour_float'))	
	filtered_outputevents = filtered_outputevents.with_columns(pl.col('value').alias('valuenum'))
	filtered_outputevents = filtered_outputevents.with_columns(pl.lit('').alias('value'))
	cols = ['subject_id', 'itemid', 'hour_float', 'value', 'valuenum', 'mortality_tf']
	all_events = pl.concat([filtered_labevents.select(cols), filtered_chartevents.select(cols), filtered_outputevents.select(cols)],\
							how = 'vertical')
	logger = get_dagster_logger()
	logger.info(all_events.head())
	
	cont_vars = all_events.groupby(['subject_id', 'hour_float', 'itemid']).agg(pl.col("valuenum").max())  
	dis_vars = all_events.groupby(['subject_id', 'hour_float', 'itemid']).agg(pl.col("value").is_not_null().last())  
		# Tokenize cont vars
	cont_vars = filtered_labevents.filter(pl.col("valuenum").is_not_null())
	cont_vars = cont_vars.with_columns(pl.col("valuenum").mean().over("itemid").suffix("_mean"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").std().over("itemid").suffix("_std"))
	cont_vars = cont_vars.with_columns(((pl.col("valuenum") - pl.col("valuenum_mean")) / pl.col("valuenum_std"))\
						 .over("itemid").suffix("_norm"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.1).over("itemid").suffix("_0"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.2).over("itemid").suffix("_1"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.3).over("itemid").suffix("_2"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.4).over("itemid").suffix("_3"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.5).over("itemid").suffix("_4"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.6).over("itemid").suffix("_5"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.7).over("itemid").suffix("_6"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.8).over("itemid").suffix("_7"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.9).over("itemid").suffix("_8"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(1  ).over("itemid").suffix("_9"))
	cont_vars = cont_vars.with_columns(
                    pl.when(pl.col("valuenum") < pl.col("valuenum_0"))\
                      .then(pl.lit(0))\
                      .when((pl.col("valuenum") < pl.col("valuenum_1")) & (pl.col("valuenum") >= pl.col("valuenum_0")))\
                      .then(pl.lit(1))\
                      .when((pl.col("valuenum") < pl.col("valuenum_2")) & (pl.col("valuenum") >= pl.col("valuenum_1")))\
                      .then(pl.lit(2))\
                      .when((pl.col("valuenum") < pl.col("valuenum_3")) & (pl.col("valuenum") >= pl.col("valuenum_2")))\
                      .then(pl.lit(3))\
                      .when((pl.col("valuenum") < pl.col("valuenum_4")) & (pl.col("valuenum") >= pl.col("valuenum_3")))\
                      .then(pl.lit(4))\
                      .when((pl.col("valuenum") < pl.col("valuenum_5")) & (pl.col("valuenum") >= pl.col("valuenum_4")))\
                      .then(pl.lit(5))\
                      .when((pl.col("valuenum") < pl.col("valuenum_6")) & (pl.col("valuenum") >= pl.col("valuenum_5")))\
                      .then(pl.lit(6))\
                      .when((pl.col("valuenum") < pl.col("valuenum_7")) & (pl.col("valuenum") >= pl.col("valuenum_6")))\
                      .then(pl.lit(7))\
                      .when((pl.col("valuenum") < pl.col("valuenum_8")) & (pl.col("valuenum") >= pl.col("valuenum_7")))\
                      .then(pl.lit(8))\
                      .when((pl.col("valuenum") < pl.col("valuenum_9")) & (pl.col("valuenum") >= pl.col("valuenum_8")))\
                      .then(pl.lit(9))\
                      .otherwise(pl.lit(10)).alias("10bins_idx"))
	cont_vars = cont_vars.with_columns(pl.concat_str("itemid", pl.lit("_"), "10bins_idx").alias("itemid_tokens"))

	# Tokenize discrete vars
	disc_vars = filtered_labevents.filter(pl.col("valuenum").is_null())
	disc_vars = disc_vars.with_columns(pl.col("value").str.to_lowercase().alias("value_lw"))
	disc_vars = disc_vars.with_columns(pl.col("value_lw").str.replace(" ", "_").alias("value_lw"))
	disc_vars = disc_vars.with_columns(pl.concat_list(["itemid", "value_lw"]).arr.join("_").alias("itemid_tokens"))

	# Extract token itemids
	logger.info(sorted(disc_vars.columns))
	logger.info(sorted(cont_vars.columns))
	all_vars = pl.concat([disc_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"]), 
						  cont_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"])], how="vertical")
	logger.info(all_vars.head())
	logger.info(all_vars.shape)
	total_tokens = all_vars.groupby(pl.col('itemid_tokens')).sum()
	logger.info(f'total token {total_tokens}')
	return all_vars

@asset(group_name="tokenize_itemid_by_episode")
def tokenize_itemid_labs(filtered_labevents):
	logger = get_dagster_logger()
	filtered_labevents = filtered_labevents.with_columns(pl.col('hour').cast(pl.Float64).alias('hour_float'))	

	# Tokenize cont vars
	cont_vars = filtered_labevents.filter(pl.col("valuenum").is_not_null())
	cont_vars = cont_vars.with_columns(pl.col("valuenum").mean().over("itemid").suffix("_mean"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").std().over("itemid").suffix("_std"))
	cont_vars = cont_vars.with_columns(((pl.col("valuenum") - pl.col("valuenum_mean")) / pl.col("valuenum_std"))\
						 .over("itemid").suffix("_norm"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.1).over("itemid").suffix("_0"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.2).over("itemid").suffix("_1"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.3).over("itemid").suffix("_2"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.4).over("itemid").suffix("_3"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.5).over("itemid").suffix("_4"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.6).over("itemid").suffix("_5"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.7).over("itemid").suffix("_6"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.8).over("itemid").suffix("_7"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.9).over("itemid").suffix("_8"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(1  ).over("itemid").suffix("_9"))
	cont_vars = cont_vars.with_columns(
                    pl.when(pl.col("valuenum") < pl.col("valuenum_0"))\
                      .then(pl.lit(0))\
                      .when((pl.col("valuenum") < pl.col("valuenum_1")) & (pl.col("valuenum") >= pl.col("valuenum_0")))\
                      .then(pl.lit(1))\
                      .when((pl.col("valuenum") < pl.col("valuenum_2")) & (pl.col("valuenum") >= pl.col("valuenum_1")))\
                      .then(pl.lit(2))\
                      .when((pl.col("valuenum") < pl.col("valuenum_3")) & (pl.col("valuenum") >= pl.col("valuenum_2")))\
                      .then(pl.lit(3))\
                      .when((pl.col("valuenum") < pl.col("valuenum_4")) & (pl.col("valuenum") >= pl.col("valuenum_3")))\
                      .then(pl.lit(4))\
                      .when((pl.col("valuenum") < pl.col("valuenum_5")) & (pl.col("valuenum") >= pl.col("valuenum_4")))\
                      .then(pl.lit(5))\
                      .when((pl.col("valuenum") < pl.col("valuenum_6")) & (pl.col("valuenum") >= pl.col("valuenum_5")))\
                      .then(pl.lit(6))\
                      .when((pl.col("valuenum") < pl.col("valuenum_7")) & (pl.col("valuenum") >= pl.col("valuenum_6")))\
                      .then(pl.lit(7))\
                      .when((pl.col("valuenum") < pl.col("valuenum_8")) & (pl.col("valuenum") >= pl.col("valuenum_7")))\
                      .then(pl.lit(8))\
                      .when((pl.col("valuenum") < pl.col("valuenum_9")) & (pl.col("valuenum") >= pl.col("valuenum_8")))\
                      .then(pl.lit(9))\
                      .otherwise(pl.lit(10)).alias("10bins_idx"))
	cont_vars = cont_vars.with_columns(pl.concat_str("itemid", pl.lit("_"), "10bins_idx").alias("itemid_tokens"))

	# Tokenize discrete vars
	disc_vars = filtered_labevents.filter(pl.col("valuenum").is_null())
	disc_vars = disc_vars.with_columns(pl.col("value").str.to_lowercase().alias("value_lw"))
	disc_vars = disc_vars.with_columns(pl.col("value_lw").str.replace(" ", "_").alias("value_lw"))
	disc_vars = disc_vars.with_columns(pl.concat_list(["itemid", "value_lw"]).arr.join("_").alias("itemid_tokens"))

	# Extract token itemids
	logger.info(sorted(disc_vars.columns))
	logger.info(sorted(cont_vars.columns))
	all_vars = pl.concat([disc_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"]), 
						  cont_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"])], how="vertical")
	return all_vars


@asset(group_name="tokenize_itemid_by_episode")
def tokenize_itemid_charts(filtered_chartevents, filtered_patients):
	logger = get_dagster_logger()
	filtered_chartevents =  filtered_chartevents.with_columns(pl.col('hour').cast(pl.Float64).alias('hour_float'))	
	filtered_chartevents = filtered_chartevents.lazy().join(filtered_patients.lazy(), on=['subject_id'])
	filtered_chartevents = filtered_chartevents.collect()

	# Tokenize cont vars
	cont_vars = filtered_chartevents.filter(pl.col("valuenum").is_not_null())
	cont_vars = cont_vars.with_columns(pl.col("valuenum").mean().over("itemid").suffix("_mean"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").std().over("itemid").suffix("_std"))
	cont_vars = cont_vars.with_columns(((pl.col("valuenum") - pl.col("valuenum_mean")) / pl.col("valuenum_std"))\
						 .over("itemid").suffix("_norm"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.1).over("itemid").suffix("_0"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.2).over("itemid").suffix("_1"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.3).over("itemid").suffix("_2"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.4).over("itemid").suffix("_3"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.5).over("itemid").suffix("_4"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.6).over("itemid").suffix("_5"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.7).over("itemid").suffix("_6"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.8).over("itemid").suffix("_7"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.9).over("itemid").suffix("_8"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(1  ).over("itemid").suffix("_9"))
	cont_vars = cont_vars.with_columns(
                    pl.when(pl.col("valuenum") < pl.col("valuenum_0"))\
                      .then(pl.lit(0))\
                      .when((pl.col("valuenum") < pl.col("valuenum_1")) & (pl.col("valuenum") >= pl.col("valuenum_0")))\
                      .then(pl.lit(1))\
                      .when((pl.col("valuenum") < pl.col("valuenum_2")) & (pl.col("valuenum") >= pl.col("valuenum_1")))\
                      .then(pl.lit(2))\
                      .when((pl.col("valuenum") < pl.col("valuenum_3")) & (pl.col("valuenum") >= pl.col("valuenum_2")))\
                      .then(pl.lit(3))\
                      .when((pl.col("valuenum") < pl.col("valuenum_4")) & (pl.col("valuenum") >= pl.col("valuenum_3")))\
                      .then(pl.lit(4))\
                      .when((pl.col("valuenum") < pl.col("valuenum_5")) & (pl.col("valuenum") >= pl.col("valuenum_4")))\
                      .then(pl.lit(5))\
                      .when((pl.col("valuenum") < pl.col("valuenum_6")) & (pl.col("valuenum") >= pl.col("valuenum_5")))\
                      .then(pl.lit(6))\
                      .when((pl.col("valuenum") < pl.col("valuenum_7")) & (pl.col("valuenum") >= pl.col("valuenum_6")))\
                      .then(pl.lit(7))\
                      .when((pl.col("valuenum") < pl.col("valuenum_8")) & (pl.col("valuenum") >= pl.col("valuenum_7")))\
                      .then(pl.lit(8))\
                      .when((pl.col("valuenum") < pl.col("valuenum_9")) & (pl.col("valuenum") >= pl.col("valuenum_8")))\
                      .then(pl.lit(9))\
                      .otherwise(pl.lit(10)).alias("10bins_idx"))
	cont_vars = cont_vars.with_columns(pl.concat_str("itemid", pl.lit("_"), "10bins_idx").alias("itemid_tokens"))


	# Tokenize discrete vars
	disc_vars = filtered_chartevents.filter(pl.col("valuenum").is_null())
	disc_vars = disc_vars.with_columns(pl.col("value").str.to_lowercase().alias("value_lw"))
	disc_vars = disc_vars.with_columns(pl.col("value_lw").str.replace(" ", "_").alias("value_lw"))
	disc_vars = disc_vars.with_columns(pl.concat_list(["itemid", "value_lw"]).arr.join("_").alias("itemid_tokens"))

	# Extract token itemids
	logger.info(sorted(disc_vars.columns))
	logger.info(sorted(cont_vars.columns))
	all_vars = pl.concat([disc_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"]), 
						  cont_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"])], how="vertical")
	logger.info(f'Tokenize chart events head: {all_vars.head()}')
	logger.info(f'Tokenize chart event tail: {all_vars.tail()}')
	logger.info(f'Tokenize chart event shape: {all_vars.shape}')

	return all_vars

@asset(group_name="tokenize_itemid_by_episode")
def tokenize_itemid_outputs(filtered_outputevents):
	logger = get_dagster_logger()
	filtered_outputevents = filtered_outputevents.with_columns(pl.col("value").alias("valuenum"))
	filtered_outputevents = filtered_outputevents.with_columns(pl.col('hour').cast(pl.Float64).alias('hour_float'))
	# Tokenize cont vars
	cont_vars = filtered_outputevents.filter(pl.col("valuenum").is_not_null())
	cont_vars = cont_vars.with_columns(pl.col("valuenum").mean().over("itemid").suffix("_mean"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").std().over("itemid").suffix("_std"))
	cont_vars = cont_vars.with_columns(((pl.col("valuenum") - pl.col("valuenum_mean")) / pl.col("valuenum_std"))\
						 .over("itemid").suffix("_norm"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.1).over("itemid").suffix("_0"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.2).over("itemid").suffix("_1"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.3).over("itemid").suffix("_2"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.4).over("itemid").suffix("_3"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.5).over("itemid").suffix("_4"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.6).over("itemid").suffix("_5"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.7).over("itemid").suffix("_6"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.8).over("itemid").suffix("_7"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(0.9).over("itemid").suffix("_8"))
	cont_vars = cont_vars.with_columns(pl.col("valuenum").quantile(1  ).over("itemid").suffix("_9"))
	cont_vars = cont_vars.with_columns(
                    pl.when(pl.col("valuenum") < pl.col("valuenum_0"))\
                      .then(pl.lit(0))\
                      .when((pl.col("valuenum") < pl.col("valuenum_1")) & (pl.col("valuenum") >= pl.col("valuenum_0")))\
                      .then(pl.lit(1))\
                      .when((pl.col("valuenum") < pl.col("valuenum_2")) & (pl.col("valuenum") >= pl.col("valuenum_1")))\
                      .then(pl.lit(2))\
                      .when((pl.col("valuenum") < pl.col("valuenum_3")) & (pl.col("valuenum") >= pl.col("valuenum_2")))\
                      .then(pl.lit(3))\
                      .when((pl.col("valuenum") < pl.col("valuenum_4")) & (pl.col("valuenum") >= pl.col("valuenum_3")))\
                      .then(pl.lit(4))\
                      .when((pl.col("valuenum") < pl.col("valuenum_5")) & (pl.col("valuenum") >= pl.col("valuenum_4")))\
                      .then(pl.lit(5))\
                      .when((pl.col("valuenum") < pl.col("valuenum_6")) & (pl.col("valuenum") >= pl.col("valuenum_5")))\
                      .then(pl.lit(6))\
                      .when((pl.col("valuenum") < pl.col("valuenum_7")) & (pl.col("valuenum") >= pl.col("valuenum_6")))\
                      .then(pl.lit(7))\
                      .when((pl.col("valuenum") < pl.col("valuenum_8")) & (pl.col("valuenum") >= pl.col("valuenum_7")))\
                      .then(pl.lit(8))\
                      .when((pl.col("valuenum") < pl.col("valuenum_9")) & (pl.col("valuenum") >= pl.col("valuenum_8")))\
                      .then(pl.lit(9))\
                      .otherwise(pl.lit(10)).alias("10bins_idx"))
	cont_vars = cont_vars.with_columns(pl.concat_str("itemid", pl.lit("_"), "10bins_idx").alias("itemid_tokens"))
	cont_vars = cont_vars.select(["subject_id", "hour_float", "itemid_tokens", "mortality_tf"])
	return cont_vars



@asset(group_name="subject_to_episode_allevents")
def allevents_by_episode(config: MyAssetConfig, tokenize_itemid_charts, tokenize_itemid_labs, tokenize_itemid_outputs) -> Output:
	logger = get_dagster_logger()
	df = pl.concat([tokenize_itemid_charts, tokenize_itemid_labs, tokenize_itemid_outputs], how='vertical')
	bins = list(range(config.time_h+1))
	df = all_events
	df = df.with_columns(pl.cut(df.select(pl.col('hour_float')).to_series(), bins=bins))
	logger.info(df.head())
	logger.info(df.tail())


	df = df.with_columns(pl.col("itemid_tokens").alias("itemid"))
	unique_items = list(set(df.select(pl.col('itemid_tokens')).to_series().to_list()))
	logger.info(unique_items[:5])
	item2index =  {item: index for index, item in enumerate(unique_items)}
	index2item =  {index: item for index, item in enumerate(unique_items)}
	logger.info(len(item2index))
	df = df.with_columns(pl.col('itemid_tokens').map_dict(item2index).alias('itemidx'))

	# Category by hour and deduplicate the itemidx
	df_mortality = df.groupby(['subject_id', 'category']).agg(pl.col('mortality_tf'))
	df_item = df.groupby(['subject_id', 'category']).agg(pl.col('itemidx').unique())
	both = df_item.join(df_mortality, on=['subject_id', 'category'])

	first = both.groupby(['subject_id']).agg(pl.col('itemidx'))
	second= both.groupby(['subject_id']).agg(pl.col('mortality_tf').arr.contains(True).any())
	df = pl.concat([first.select(['subject_id', 'itemidx']), second.select('mortality_tf')], how='horizontal')
	logger.info(df.head())
	return Output(  
        value=df,
        metadata={
            "shape": f'{df.shape}', 
            "schema": f'{df.schema}',
			
        }
)


