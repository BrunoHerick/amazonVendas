# coding: utf-8
import pandas as pd
import matplotlib.pyplot as grafico
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#OBJECTIVE: predict the likelihood of a new order being rejected

dados = pd.read_csv("/home/brunohelghast/PROFISSIONAL/PYTHON/SCIKIT_LEARN/amazom/amazom_dados.csv")
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

dados_previsores[:,0] = label_encoder_buyer.fit_transform(dados_previsores[:,0].astype(str))
dados_previsores[:,1] = label_encoder_ship_city.fit_transform(dados_previsores[:,1].astype(str))
dados_previsores[:,2] = label_encoder_ship_state.fit_transform(dados_previsores[:,2].astype(str))
dados_previsores[:,3] = label_encoder_sku.fit_transform(dados_previsores[:,3].astype(str))
dados_previsores[:,4] = label_encoder_description.fit_transform(dados_previsores[:,4].astype(str))
dados_previsores[:,5] = label_encoder_quantity.fit_transform(dados_previsores[:,5].astype(str))
dados_previsores[:,6] = label_encoder_item_total.fit_transform(dados_previsores[:,6].astype(str))
dados_previsores[:,7] = label_encoder_shipping_fee.fit_transform(dados_previsores[:,7].astype(str))
dados_previsores[:,8] = label_encoder_cod.fit_transform(dados_previsores[:,8].astype(str))

arvore_dados_previsores = DecisionTreeClassifier(criterion="entropy")
arvore_dados_previsores.fit(dados_previsores, dados_previstos)

previsoes = arvore_dados_previsores.predict([[46,24,25,45,30,2,30,50,0]])
print(previsoes)

# Produto mais cancelado: Women's Set of 5 Multicolor Pure Leather Single Lipstick Cases with Mirror, Handy and Compact Handcrafted Shantiniketan Block Printed Jewelry Boxes (3 pedidos cancelados)
# 63.64 dos pedidos teriam uma taxa de entrega maritima de $84.96 