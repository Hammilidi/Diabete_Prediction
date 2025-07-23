import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

#Méthode qui supprime toutes les valeurs abérrentes
def clean_data_1(df):
    df = df.rename(columns={'Unnamed: 0': 'index_tab'})
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
    df = df.rename(columns={'Unnamed: 0': 'index_tab'})
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

def make_cluster(data, k ,colonnes_cluster , scaler=True):

    if scaler== True :
        X= data[colonnes_cluster]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


    # Créez le modèle KMeans et entraînez-le sur les données
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)

    # Récupérez les labels (attribution des clusters pour chaque ligne)
    labels = kmeans.labels_

    # Ajoutez les labels au DataFrame original (nouvelle colonne 'cluster')
    data['cluster'] = labels

    return data