from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Dados fictícios: [salário, depósito mensal, histórico (0=limpo, 1=atrasado), dependentes, valor solicitado]
dados = [
    [5000, 2500, 0, 1, 15000],  # Aprovado
    [5000, 2500, 1, 0, 15000],  # Rejeitado
    [5000, 2500, 0, 1, 15000],   # Aprovado
    [5000, 2500, 1, 2, 10000],  # Rejeitado
    [5000, 2500, 0, 3, 10000],  # Aprovado
    [5000, 2500, 1, 0, 10000],  # Rejeitado
    #///////////////////////////
    [5000, 5000, 0, 3, 5000],  # Aprovado
    [5000, 5000, 1, 0, 5000],  # Rejeitado
    [5000, 5000, 1, 5, 5000],   # Aprovado
    [5000, 5000, 1, 2, 1000],  # Rejeitado
    [5000, 5000, 0, 5, 1000],  # Aprovado
    [5000, 5000, 1, 3, 1000],  # Rejeitado
    #///////////////////////////
    [1000, 2500, 0, 0, 5000],  # Aprovado
    [1000, 2500, 1, 0, 5000],  # Rejeitado
    [1000, 2500, 0, 1, 5000],   # Aprovado
    [1000, 2500, 1, 2, 1000],  # Rejeitado
    [1000, 2500, 0, 0, 1000],  # Aprovado
    [1000, 2500, 1, 0, 1000],  # Rejeitado
    #///////////////////////////
    [0, 500, 0, 1, 1000],  # Aprovado
    [0, 500, 1, 0, 1000],  # Rejeitado
    [0, 500, 0, 1, 1000],   # Aprovado
    [500, 2500, 1, 2, 5000],  # Rejeitado
    [500, 2500, 1, 0, 1000],  # Aprovado
    [500, 2500, 1, 0, 5000],  # Rejeitado
    #///////////////////////////
    [5000, 500, 0, 0, 5000],  # Aprovado
    [5000, 500, 1, 1, 5000],  # Rejeitado
    [5000, 500, 0, 4, 5000],   # Aprovado
    [5000, 2500, 1, 5, 500],  # Rejeitado
    [5000, 2500, 1, 0, 500],  # Aprovado
    [5000, 2500, 0, 5, 500],  # Rejeitado
    #///////////////////////////
    [5000, 10000, 0, 4, 1000],  # Aprovado
    [5000, 10000, 1, 0, 1000],  # Rejeitado
    [5000, 10000, 0, 3, 1000],   # Aprovado
    [5000, 10000, 1, 3, 500],  # Rejeitado
    [5000, 10000, 0, 3, 500],  # Aprovado
    [5000, 10000, 1, 5, 500],  # Rejeitado
     
]

resposta = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,] # Respostas(1=aprovado 0=N/aprovado)

# Normalizando os dados
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(dados)

# Divisao em treino e teste (80%/20%)
x_treino, x_teste, y_treino, y_teste = train_test_split(dados_normalizados, resposta, test_size=0.2, random_state=42)

# Configurando os hiperparâmetros a testar
parametros = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}

# Busca pelos melhores hiperparâmetros com validação cruzada
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), parametros, cv=5)
grid_search.fit(x_treino, y_treino)

# Melhor modelo encontrado
melhor_modelo = grid_search.best_estimator_
print("Melhores parâmetros:", grid_search.best_params_)

# Fazer previsões com o melhor modelo
previsoes = melhor_modelo.predict(x_teste)

# Avaliar o modelo
precisao = accuracy_score(y_teste, previsoes)
print("Precisão com o melhor modelo:", precisao)


# Dados de vários candidatos [salário, depósito mensal, histórico (0=limpo, 1=atrasado), dependentes, valor solicitado

candidatos = [
    [4000, 1500, 1, 2, 8000],  # Candidato 1
    [7000, 3000, 0, 1, 10000],  # Candidato 2
    [2000, 1000, 1, 4, 5000],   # Candidato 3
    [0, 5000, 0, 0, 1000],   # Candidato 4
    [10000, 9000, 1, 7, 5000],  # Candidato 5
    [0, 100, 0, 0, 700],    # Candidato 6
    [5000, 5000, 1, 10, 100000],    # Candidato 7
    [1000, 500, 1, 0, 50],  # Candidato 8
    [7000, 5000, 0, 2, 2500],    # Candidato 9
     [10000, 5000, 1, 0, 2500]    # Candidato 10
]

# Fazer previsões
resultados = melhor_modelo.predict(candidatos)

# Exibir resultados
for i, resultado in enumerate(resultados):
    print(f"Candidato {i + 1}: {'Aprovado' if resultado == 1 else 'Rejeitado'}")

# Grafico
plt.figure(figsize=(8, 4))  # Define o tamanho do gráfico
plot_tree(
    melhor_modelo,  # Modelo treinado
    feature_names=["Salário", "Depósito Mensal", "Histórico", "Dependentes", "Valor Solicitado"],  # Nomes das variáveis
    class_names=["Rejeitado", "Aprovado"],  # Classes possíveis
    filled=True  # Preenche os nós com cores baseadas na classe predominante
)
plt.show()
print('--------------------')

# Previsões e valores reais
y_real = y_teste
y_previsto = previsoes

# Matriz de confusão
matriz = confusion_matrix(y_real, y_previsto)
print("Matriz de Confusão:")
print(matriz)

# Relatório detalhado (Precision, Recall, F1-score)
relatorio = classification_report(y_real, y_previsto, target_names=["Rejeitado", "Aprovado"])
print("\nRelatório de Desempenho:")
print(relatorio)
