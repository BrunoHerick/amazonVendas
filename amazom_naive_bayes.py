#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as grafico
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# OBJECTIVE: predict the likelihood of a new order being rejected
dados = pd.read_csv("/home/brunohelghast/PROFISSIONAL/PYTHON/SCIKIT_LEARN/amazom/amazom_dados.csv")
dados.fillna("---",inplace=True)

dados_previsores = dados.iloc[:,0:9].values
dados_previstos = dados.iloc[:,9].values
aux1 = dados.iloc[:,0:9].values
aux2 = dados.iloc[:,9].values

encoder_buyer = LabelEncoder()
encoder_ship_city = LabelEncoder()
encoder_ship_state = LabelEncoder()
encoder_sku = LabelEncoder()
encoder_description = LabelEncoder()
encoder_quantity = LabelEncoder()
encoder_item_total = LabelEncoder()
encoder_shipping_fee = LabelEncoder()
encoder_cod = LabelEncoder()

dados_previsores[:,0] = encoder_buyer.fit_transform(dados_previsores[:,0].astype(str))
dados_previsores[:,1] = encoder_ship_city.fit_transform(dados_previsores[:,1].astype(str))
dados_previsores[:,2] = encoder_ship_state.fit_transform(dados_previsores[:,2].astype(str))
dados_previsores[:,3] = encoder_sku.fit_transform(dados_previsores[:,3].astype(str))
dados_previsores[:,4] = encoder_description.fit_transform(dados_previsores[:,4].astype(str))
dados_previsores[:,5] = encoder_quantity.fit_transform(dados_previsores[:,5].astype(str))
dados_previsores[:,6] = encoder_item_total.fit_transform(dados_previsores[:,6].astype(str))
dados_previsores[:,7] = encoder_shipping_fee.fit_transform(dados_previsores[:,7].astype(str))
dados_previsores[:,8] = encoder_cod.fit_transform(dados_previsores[:,8].astype(str))

for i in range(0, len(dados_previsores[:,0]),1):
    print(dados_previsores[i,:],aux1[i,:])

naive_situacao_pedido = GaussianNB()
naive_situacao_pedido.fit(dados_previsores, dados_previstos)# treinar
 
# combinacoes = [146,73,26,53,60,3,18,13,1]
preverSituacao = naive_situacao_pedido.predict([[46,24,25,45,30,2,30,50,0]])
print(preverSituacao)

# Produto mais cancelado: Women's Set of 5 Multicolor Pure Leather Single Lipstick Cases with Mirror, Handy and Compact Handcrafted Shantiniketan Block Printed Jewelry Boxes (3 pedidos cancelados)
# 63.64 dos pedidos teriam uma taxa de entrega maritima de $84.96