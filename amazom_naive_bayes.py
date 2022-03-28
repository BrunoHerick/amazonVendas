#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as grafico
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# OBJECTIVE: predict the likelihood of a new order being rejected
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

for i in range(0, len(dados_previsores[:,0]),1):
    print(dados_previsores[i,:],aux1[i,:])

naive_situacao_pedido = GaussianNB()
naive_situacao_pedido.fit(dados_previsores, dados_previstos)# treinar

# combinacoes = [146,73,26,53,60,3,18,13,1]
preverSituacao = naive_situacao_pedido.predict([[46,24,25,45,30,2,30,50,0]])
print(preverSituacao)