#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as grafico
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#OBJECTIVE: predict the likelihood of a new order being rejected

dados = pd.read_csv("/home/bhs/PROFISSIONAL/PYTHON/SCIKIT_LEARN/amazom_dados.csv")
dados.fillna("---",inplace=True)

dados_previsores = dados.iloc[:,0:9].values
dados_previstos = dados.iloc[:,9].values
aux1 = dados.iloc[:,0:9].values
aux2 = dados.iloc[:,9].values

label_encoder_buyer = LabelEncoder()
label_encoder_ship_city = LabelEncoder()
label_encoder_ship_state = LabelEncoder()
label_encoder_sku = LabelEncoder()
label_encoder_description = LabelEncoder()
label_encoder_quantity = LabelEncoder()
label_encoder_item_total = LabelEncoder()
label_encoder_shipping_fee = LabelEncoder()
label_encoder_cod = LabelEncoder()

dados_previsores[:,0] = label_encoder_buyer.fit_transform(dados_previsores[:,0])
dados_previsores[:,1] = label_encoder_ship_city.fit_transform(dados_previsores[:,1])
dados_previsores[:,2] = label_encoder_ship_state.fit_transform(dados_previsores[:,2])
dados_previsores[:,3] = label_encoder_sku.fit_transform(dados_previsores[:,3])
dados_previsores[:,4] = label_encoder_description.fit_transform(dados_previsores[:,4])
dados_previsores[:,5] = label_encoder_quantity.fit_transform(dados_previsores[:,5])
dados_previsores[:,6] = label_encoder_item_total.fit_transform(dados_previsores[:,6])
dados_previsores[:,7] = label_encoder_shipping_fee.fit_transform(dados_previsores[:,7])
dados_previsores[:,8] = label_encoder_cod.fit_transform(dados_previsores[:,8])

arvore_dados_previsores = DecisionTreeClassifier(criterion="entropy")
arvore_dados_previsores.fit(dados_previsores, dados_previstos)

#print(arvore_situacao_alunos.feature_importances_) --> [0.227 0.133 0.37 0.267 0. ] = ganhos de informação, matematica tem o maior ganho de informacao com 0.37
""" MOSTRAR ARVORE 
previsores = ["PORTUGUES","CIENCIA","MATEMATICA","HISTORIA","GEOGRAFIA","SITUACAO"]
figura, eixos = grafico.subplots(nrows=1, ncols=1,figsize=[10,10])
tree.plot_tree(arvore_situacao_alunos, feature_names=previsores, class_names=arvore_situacao_alunos.classes, filled=True) """

previsoes = arvore_dados_previsores.predict([[46,24,25,45,30,2,30,50,0]])
print(previsoes)