import polars as pl

patients = pl.read_parquet('data/patients.parquet')
cols=patients.columns
dtypes=patients.dtypes
print('Table patients {')
for i in zip(cols, dtypes):
	print(f'{i[0]} {i[1]}')
print('}')
print('\n')

chartevents = pl.read_parquet('data/chartevents.parquet')
cols = chartevents.columns
dtypes= chartevents.dtypes
print('Table chartevents {')
for i in zip(cols, dtypes):
	print(f'{i[0]} {i[1]}')
print('}')
print('\n')

icustays = pl.read_parquet('data/icustays.parquet')
cols = icustays.columns
dtypes= icustays.dtypes
print('Table icustays {')
for i in zip(cols, dtypes):
	print(f'{i[0]} {i[1]}')
print('}')
print('\n\n')


admissions = pl.read_parquet('data/admissions.parquet')
cols = admissions.columns
dtypes = admissions.dtypes
print('Table admissions {')
for i in zip(cols, dtypes):
	print(f'{i[0]} {i[1]}')
print('}')
print('\n\n')


labevents = pl.read_parquet('data/labevents.parquet')
cols = labevents.columns
dtypes = labevents.dtypes
print('Table labevents {')
for i in zip(cols, dtypes):
	print(f'{i[0]} {i[1]}')
print('}')
print('\n\n')


outputevents = pl.read_parquet('data/outputevents.parquet')
cols = outputevents.columns
dtypes = outputevents.dtypes
print('Table outputevents {')
for i in zip(cols, dtypes):
	print(f'{i[0]} {i[1]}')
print('}')
print('\n\n')