import pandas as pd
# tabela = pd.read_csv("clientes.csv") -> Substitua pelo csv da sua empresa
display(tabela.info())

# A IA só aprende com números, logo, temos que tirar todo texto usando LabelEncoder

from sklearn.preprocessing import LabelEncoder
codificador_profissao = LabelEncoder()
tabela["profissao"] = codificador_profissao.fit_transform(tabela["profissao"])

codificador_credito = LabelEncoder()
tabela["mix_credito"] = codificador_credito.fit_transform(tabela["mix_credito"])

codificador_pagamento = LabelEncoder()
tabela["comportamento_pagamento"] = codificador_pagamento.fit_transform(tabela["comportameento_pagamento"])

display(tabela.info()

# Y é o que eu quero prever
y = tabela["score_credito"]

# X são as colunas da base de dados que vou usar para fazer a previsão
x = tabela.drop(columns=["score_credito", "id_cliente"])


from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2)

# Passo 3 - Treinar a Inteligência Artificial -> 
# Criar o modelo: Nota de crédito: Boa, Ok, Ruim

# Arvore de Decisão -> RandomForest
# Nearest Neighbors -> KNN -> Vizinhos Próximos

# importar a IA (Inteligencia Artificial)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Criar a IA
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

#Treinar a IA
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)


previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_nn = modelo_knn.predict(x_teste)

# Acurácia (Quanto ela é efetiva)
from sklearn.metrics import accuracy_score
display(accuracy_score(y_teste, previsao_arvoredecisao))
display(accuracy_score(y_teste, previsao_nn))

# Importar novos clientes para fazer a previsão
tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
tabela_novos_clientes["profissao"] = codificador_profissao.transform(
    tabela_novos_clientes["profissao"])


tabela_novos_clientes["mix_credito"] = codificador_credito.transform(
    tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"] = codificador_pagamento.transform(
    tabela_novos_clientes["comportamento_pagamento"])
display(tabela_novos_clientes)

nova_previsao = modelo_arvoredecisao.predict(tabela_novos_clientes)
display(nova_previsao)
