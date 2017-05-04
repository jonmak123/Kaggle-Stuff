# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:49:17 2017

@author: J-Mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##### Plotly modules #####
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
##########################

##### Map utils #####
from geopy.geocoders import Nominatim
#####################

plt.style.use('seaborn-bright')

#https://gist.github.com/rogerallen/1583593
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

#df = pd.read_csv('database.csv', low_memory=False)
#
#df['Full Date'] = pd.to_datetime((str(15) + '-' + df['Month'] + '-' + df['Year'].astype(str)), format='%d-%B-%Y')
#df['Weapon Type'] = df['Weapon'].apply(lambda x: 'Gun' if x in ['Rifle', 'Firearm', 'Shotgun', 'Handgun', 'Gun'] else 'Other')
#df.loc[df['Perpetrator Age']==" ", 'Perpetrator Age'] = 0
#df['Perpetrator Age'] = df['Perpetrator Age'].astype(int)
#df['state'] = df['State'].map(us_state_abbrev)

def national_total(df):
    df1 = df.groupby(['Full Date'], as_index=False)['Incident'].sum()
    x = df1['Full Date']
    y = df1['Incident']
    
    f1, ax1 = plt.subplots()
    ax1 = plt.plot(x, y)
    f1.suptitle('National Homicide Counts')
    #plot([Scatter(x=x, y=y)])

def state_total(df):
    df2 = df.groupby(['Full Date', 'Year', 'State', 'state'], as_index=False)['Incident'].sum()

    f2, ax2 = plt.subplots()
    for state in df2['State'].unique():
        x = df2[df2['State']==state]['Full Date']
        y = df2[df2['State']==state]['Incident']
        ax2.plot(x, y, label=state)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def state_total_py(df):
    df2 = df.groupby(['Full Date', 'Year', 'State', 'state'], as_index=False)['Incident'].sum()
    py_data = []
    for state in df2['State'].unique():
        x = df2[df2['State']==state]['Full Date']
        y = df2[df2['State']==state]['Incident']
    #    ax2.plot(x, y, label=state)
        trace = go.Scatter(x=x, y=y, mode='lines', name=state)
        py_data.append(trace)
    layout = dict(title='Homicide Count by State', xaxis=dict(title='Year'))
    fig = dict(data=py_data, layout=layout)
    plot(fig)

def state_weapon(df):
    df3 = df.groupby(['State', 'Weapon'])['Incident'].sum().reset_index()
    df3['pc'] = df3.groupby('State')['Incident'].apply(lambda x: 100 * x / x.sum())
    
    f3, ax3 = plt.subplots()
    sns.heatmap(df3.pivot(index='State', columns='Weapon', values='pc'), square=True, ax=ax3)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

def gender_weapon(df):
    df4 = df.groupby(['Perpetrator Sex', 'Weapon'])['Incident'].sum().reset_index()
    df4['pc'] = df4.groupby('Perpetrator Sex')['Incident'].apply(lambda x: 100 * x / x.sum())
    f4, ax4 = plt.subplots()
    sns.heatmap(df4.pivot(index='Perpetrator Sex', columns='Weapon', values='pc'), square=True, ax=ax4)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    df4_1 = df.groupby(['Perpetrator Sex', 'Weapon Type'])['Incident'].sum().reset_index()
    df4_1['pc'] = df4_1.groupby('Perpetrator Sex')['Incident'].apply(lambda x: 100 * x / x.sum())
    f4_1, ax4_1 = plt.subplots()
    sns.heatmap(df4_1.pivot(index='Perpetrator Sex', columns='Weapon Type', values='pc'), square=True, ax=ax4_1)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
def gun_death(df):
    df5 = df.groupby(['State', 'state', 'Weapon Type'])['Incident'].sum().reset_index()
    df5['pc'] = df5.groupby('State')['Incident'].apply(lambda x: 100 * x / x.sum())
    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
    data = [dict(
        type = 'choropleth',
        colorscale = scl, 
        locations = df5['state'],
        z = df5['pc'],
        text = df5['State'],
        locationmode='USA-states',
        autocolorscale = True, 
        colorbar = dict(title = "% of Gun-related homocides")
        )]
    layout = Layout(geo=dict(scope = 'usa'), title = 'Gun related deaths as % of total deaths')
    fig = dict( data=data, layout=layout)
    plot(fig)