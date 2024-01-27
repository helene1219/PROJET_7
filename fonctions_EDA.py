
import pandas as pd
import seaborn as sns
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show

def data_describe(data):
    data_object={}
    data_object = [(x, data[x].dtype, 
                              data[x].isna().sum().sum(),
                              int(data[x].count())) for x in data.select_dtypes(exclude=['int', 'float'])]
    df_object = pd.DataFrame(data = data_object)
    df_object.columns=['features','dtype','nan','count']
    
    data_numeric = {}
    data_numeric = [(x, data[x].dtype, 
                               int(data[x].isna().sum().sum())/data[x].isnull().count()*100, 
                               int(data[x].count()), 
                               int(data[x].mean()), 
                               round(data[x].std(),1),
                               round(data[x].min(),1), 
                               round(data[x].max(),1)) for x in data.select_dtypes(exclude='object')]    
    df_numeric = pd.DataFrame(data = data_numeric)
    df_numeric.columns=['features','dtype','nan','count', 'mean', 'std', 'min','max']    

    return df_object, df_numeric

def nan_check(data):
    total = data.isnull().sum()
    percent_1 = data.isnull().sum()/data.isnull().count()*100
    percent_2 = (np.round(percent_1, 2))
    missing_data = pd.concat([total, percent_2], 
                             axis=1, keys=['Total', '%']).sort_values('%', ascending=False)
    return missing_data

def plot_stat(data, feature, title) : 
    with sns.color_palette("PiYG"):    
        df=data[data[feature]!="XNA"]
        ax, fig = plt.subplots(figsize=(14,6)) 
        ax = sns.countplot(y=feature, data=df, order=df[feature].value_counts(ascending=False).index)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_width()/len(df[feature]))
                    x = p.get_x() + p.get_width()
                    y = p.get_y() + p.get_height()/2
                    ax.annotate(percentage, (x, y), fontsize=14, fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        show(fig)
        
def plot_percent_target1(data, feature) : 
    df=data[data[feature]!="XNA"]
    cat_perc = df[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    with sns.color_palette("PiYG"):  
        ax, fig = plt.subplots(figsize=(14,6)) 
        ax = sns.barplot(y=feature, x='TARGET', data=cat_perc)
        ax.set_title("Répartition défaut de crédit - Target = 1")
        ax.set_xlabel("")
        ax.set_ylabel(" ")

        for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_width())
                    x = p.get_x() + p.get_width()
                    y = p.get_y() + p.get_height()/2
                    ax.annotate(percentage, (x, y), fontsize=14, fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        show()
        
#Plot distribution of one feature
def plot_distribution(data,feature, title):
    plt.figure(figsize=(20,6))

    t0 = data.loc[data['TARGET'] == 0]
    t1 = data.loc[data['TARGET'] == 1]

    
    sns.kdeplot(t0[feature].dropna(), color='purple', linewidth=4, label="TARGET = 0")
    sns.kdeplot(t1[feature].dropna(), color='C',  linewidth=4, label="TARGET = 1")
    plt.title(title)
    plt.ylabel('')
    plt.legend()
    show()  

