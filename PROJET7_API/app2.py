import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plotly.express as px
import pickle
import shap
from lightgbm import LGBMClassifier
# Change the console encoding
sys.stdout.reconfigure(encoding='utf-8')


def main() :

    
    @st.cache_data
    def load_data():
        data = pd.read_csv("application_train_sample.csv", index_col='SK_ID_CURR', encoding ='utf-8', sep ='\t')
        sample = pd.read_csv("application_train_sample.csv", encoding ='utf-8', sep ='\t')
        X_test = pd.read_csv("X_test_sample.csv", sep ='\t')
        #X_test.drop('SK_ID_CURR', axis=1, inplace=True)
        X_test.drop('Unnamed: 0', axis=1, inplace=True)
        X_test.drop('Unnamed: 0.1', axis=1, inplace=True)        
        target = data.iloc[:, -1:]

        return sample, target,data,X_test
    
    @st.cache_data
    def load_model():
        '''loading the trained model'''
        pickle_in = open('model_LGBM.pkl', 'rb') 
        clf = pickle.load(pickle_in) 
        return clf    
    
    
    @st.cache_data
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        #targets = data.TARGET.value_counts()

        #return nb_credits, rev_moy, credits_moy, targets
        return nb_credits, rev_moy, credits_moy
    
    @st.cache_data
    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client
  

    @st.cache_data
    def load_age_population(data):
        data_age = round(-(data["DAYS_BIRTH"]/365), 2)
        return data_age
    
   
   
    @st.cache_data
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income
    
    
    @st.cache_data
    def load_prediction(sample,X_test, chk_id, _clf):
        data_ID=sample[['SK_ID_CURR']]
        y_pred_lgbm_proba = clf.predict_proba(X_test)
        y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
        y_pred_lgbm_proba_df=pd.concat([y_pred_lgbm_proba_df, data_ID], axis=1)
        
        y_pred_lgbm_proba_df=y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['SK_ID_CURR']==int(chk_id)]
        prediction=y_pred_lgbm_proba_df.iat[0,1]
        
        if y_pred_lgbm_proba_df.iat[0,1]*100>10 : 
            statut="Client risqué" 
        else :
            statut="Client non risqué"
        return prediction,statut

    
    
    
    #Loading data……
    sample, target,data,X_test = load_data()
    id_client = sample[['SK_ID_CURR']].values
    clf = load_model()

            #######################################
                # SIDEBAR
             #######################################

                #Title display
    html_temp = """
    <div style="background-color: #D54773; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>

    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Customer ID selection
    st.sidebar.header("**INFORMATION GENERALE**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)


    #Loading general info
    nb_credits, rev_moy, credits_moy = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
    st.sidebar.markdown("<u>NOMBRE DE CREDIT :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Average income
    st.sidebar.markdown("<u>REVENU MOYEN DATA:</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    #AMT CREDIT
    st.sidebar.markdown("<u>MONTANT MOYEN DU CREDIT DATA :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    # HOME PAGE - MAIN CONTENT
    #######################################

    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header(" INFORMATION CLIENT SELECTIONNE ")

    if st.checkbox("AFFICHER LES INFORMATIONS SUR LE CLIENT ?",key="option1"):
        infos_client = identite_client(data, chk_id)
        st.write(" SEXE: ", infos_client["CODE_GENDER"].values[0])
        st.write(" AGE : {:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/-365)))
        st.write("SITUATION DE FAMILLE : ", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("NOMBRE D'ENFANT : {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        
        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="#D54773",bins=20)
        ax.axvline(int(-infos_client["DAYS_BIRTH"].values /365), color="black", linestyle='--')
        ax.set(title='AGE CLIENT', xlabel='AGE', ylabel='')
        st.pyplot(fig)


        st.subheader("REVENU (EN €)")
        st.write("REVENU TOTAL : {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("MONTANT DU CREDIT : {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("ANNUITE DU CREDIT : {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("MONTANT DU BIEN POUR LE CREDIT : {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))


        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="#D54773", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="black", linestyle='--')
        ax.set(title='REVENU DES CLIENTS', xlabel='REVENU (EN €)', ylabel='')
        st.pyplot(fig)

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
    
    #Customer solvability display
    st.header(" ANALYSE CREDIT DEMANDE ")
    prediction,statut = load_prediction(sample,X_test, chk_id, clf)
    st.write(" PROBABLITE DE DEFAUT : {:.0f} %".format(round(float(prediction)*100, 2)))
    st.write("STATUT DU CLIENT : ",statut)
    
    
#Feature importance / description
    if st.checkbox("AFFICHER LES RESULTATS SUR LE CLIENT ?",key="Option2"):
        nbligne=sample.loc[sample['SK_ID_CURR'] == int(chk_id)].index.item()
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.Explainer(clf)
        shap_values = explainer.shap_values(X_test)
        shap_vals = explainer(X_test)
        shap.waterfall_plot(shap_vals[nbligne][:, 0],show = False)
        st.pyplot(fig)
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)    
    
    
    
    
    
    
if __name__ == '__main__':
     main()
    