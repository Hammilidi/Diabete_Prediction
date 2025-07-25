import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Méthode qui supprime toutes les valeurs abérrentes
def clean_data_1(df):
    masque = (
        (df['Glucose'] >= 50) &
        (df['BloodPressure'] >= 40) &
        (df['BloodPressure'] <= 120) &
        (df['SkinThickness'] >= 5) &
        (df['SkinThickness'] <= 60)&
        (df['Insulin'] >= 24)&
        (df['Insulin'] <= 500)&
        (df['BMI'] >= 16)&
        (df['BMI'] <= 50)&
        (df['DiabetesPedigreeFunction'] <= 2)
    )
    df_clean = df[masque].copy()
    return df_clean

#Méthode qui suprime certaines valeurs abérrantes et qui recode 
def clean_data_2(df):
    masque = (
        (df['Glucose'] >= 50) &
        (df['BloodPressure'] >= 40) &
        (df['BloodPressure'] <= 120) &
        (df['SkinThickness'] <= 60)&
        (df['BMI'] >= 16)&
        (df['BMI'] <= 50)&
        (df['DiabetesPedigreeFunction'] <= 2)
    )
    df_clean = df[masque].copy()
    df_clean['Insuline_bool'] = np.where(df_clean['Insulin'] == 0, 0, 1)
    df_clean['SkinThickness_bool'] = np.where(df_clean['SkinThickness'] == 0, 0, 1)
    return df_clean

#Méthode qui supprime les outliers en se basant sur les quantiles
def clean_data_3(df):
    colonnes_a_traiter = ['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    df_clean = df.copy()
    for column in colonnes_a_traiter:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean

#Fonction qui nous permet de créer nos clusters
def make_cluster(df, k ,colonnes_cluster , scaler=True):
    df_cluster=df.copy()
    if scaler== True :
        X= df_cluster[colonnes_cluster]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled= df_cluster[colonnes_cluster]
   
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    
    labels = kmeans.labels_
    
    cluster_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    
    new_order = {old_label: new_label for new_label, old_label in enumerate(cluster_sizes.index)}
    
    labels = pd.Series(labels).map(new_order).values
    
    df_cluster['cluster'] = labels
    return df_cluster