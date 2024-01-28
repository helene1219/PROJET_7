# test_example.py
import pandas as pd
import pathlib

# Test pour s'assurer Nb colonnes final cohérent avec le modèle
def nbcolonne():
        data = pd.read_csv("data_work/X_test.csv", sep ='\t')
        data = data.drop('Unnamed: 0',axis=1)
        nombre_de_colonnes = data.shape[1]
        return nombre_de_colonnes

def test_nbcolonne():
    assert nbcolonne() == 112    
    
# Test pour s'assurer qu'il n'y a que des variables numériques en entrée   
def are_all_columns_numeric():
    data = pd.read_csv("data_work/X_test.csv", sep ='\t')
    data = data.drop('Unnamed: 0',axis=1)
    numeric_columns = data.select_dtypes(include=['number']).columns
    return len(numeric_columns) == len(data.columns)
    

def test_are_all_columns_numeric_for_mixed_data():
    assert are_all_columns_numeric()
    
# Test pour valider la présence de la variable TARGET dans le train
def test_variable_target_existe():
    # Supposons que df est votre DataFrame
    data = pd.read_csv('data_source/application_train.csv',nrows=50)

    # Assertion pour vérifier si "TARGET" existe dans les colonnes du DataFrame
    assert 'TARGET' in data.columns, "La variable 'TARGET' n'existe pas dans le DataFrame."
    

# Test pour valider que le modèle sérialisé est bien le répertoire de l'API
def test_model():
    # Spécifiez le chemin du répertoire
    directory_path = pathlib.Path('C:/Users/helen/Documents/OPENCLASSROOM/PROJET 7/PROJET7_API')

    # Spécifiez le nom du fichier que vous recherchez
    file_name = 'model_LGBM.pkl'

    # Construisez le chemin complet du fichier
    file_path = directory_path / file_name

    # Assertion pour vérifier si le fichier existe dans le répertoire
    assert file_path.exists(), f"Le fichier {file_name} n'existe pas dans le répertoire {directory_path}."
    

def test_fonctionsEDA():
    # Spécifiez le chemin du répertoire
    directory_path = pathlib.Path('C:/Users/helen/Documents/OPENCLASSROOM/PROJET 7')

    # Spécifiez le nom du fichier que vous recherchez
    file_name = 'fonctions_EDA.py'

    # Construisez le chemin complet du fichier
    file_path = directory_path / file_name

    # Assertion pour vérifier si le fichier existe dans le répertoire
    assert file_path.exists(), f"Le fichier {file_name} n'existe pas dans le répertoire {directory_path}."