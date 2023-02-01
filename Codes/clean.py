"""
    Codes for cleaning 'HourlyDewPointTemperature' series
    in 10 states from 2013-11-20 to 2022-11-20
"""

import pandas as pd

states = ['NC', 'SC', 'VA', 'KY', 'MD', 'NY', 'MI', 'CA', 'WA', 'WI']  # intuitive climate order
data = dict()

for state in states:
    temp = pd.read_csv('Data/'+state+'.csv', usecols=['DATE', 'REPORT_TYPE', 'HourlyDewPointTemperature'], dtype=object)
    temp['HourlyDewPointTemperature'] = pd.to_numeric(temp['HourlyDewPointTemperature'],  errors='coerce')
    temp = temp.loc[temp['REPORT_TYPE'] == 'FM-15']
    temp = temp.drop(columns=['REPORT_TYPE'])
    temp['HOUR'] = temp['DATE'].str[11:13]
    temp['DATE'] = temp['DATE'].str[:10]
    temp = temp[temp['HourlyDewPointTemperature'].notna()].astype({'HourlyDewPointTemperature': int})
    temp = temp.groupby(['DATE', 'HOUR']).first().reset_index()  # keep one observation per hour
    temp = temp.rename(columns={'HourlyDewPointTemperature': state})
    data[state] = temp

source = 'NC'
states.remove('NC')

out = data[source]
for state in states:
    out = pd.merge(out, data[state], how='left', on=['DATE', 'HOUR'])

out = out.dropna()
count = out.groupby('DATE')['DATE'].count()
keep = count[count == 24].index  # keep dates with 24 observations

out = out.loc[out['DATE'].isin(keep)]

out.to_csv('Data/dew_point_temp.csv', index=False)
