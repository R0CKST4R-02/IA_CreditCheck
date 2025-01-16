from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Dados fictícios: [salário, depósito mensal, histórico (0=limpo, 1=atrasado), dependentes, valor solicitado]
dados = [
    [5000, 2000, 0, 2, 10000],  # Aprovado
    [3000, 1500, 1, 3, 20000],  # Rejeitado
    [8000, 3000, 0, 1, 5000],   # Aprovado
    [2500, 2500, 1, 2, 15000],  # Rejeitado
    [6000, 2500, 0, 0, 20000],  # Aprovado
    [10000, 1000, 0, 4, 2500],  # Aprovado
    [1000, 0, 0, 2, 5000],      # Rejeitado
    [0, 5000, 0, 0, 10000],     # Rejeitado
    [1000, 1000, 0, 0, 5000],     # Rejeitado
    [1500, 100, 0, 0, 10],       # Aprovado
    [1000, 2000, 0, 2, 500],  # Aprovado
    [3000, 9000, 1, 5, 1000],  # Rejeitado
    [5000, 2000, 0, 1, 8000],   # Aprovado
    [2500, 2000, 1, 5, 15000],  # Rejeitado
    [6000, 2500, 0, 0, 2500],  # Aprovado
    [10000, 5000, 1, 5, 2500],  # Aprovado
    [1000, 0, 0, 2, 5000],      # Rejeitado
    [0, 5000, 0, 0, 10000],     # Rejeitado
    [1000, 200, 0, 1, 9000],     # Rejeitado
    [1500, 1500, 0, 0, 100],       # Aprovado
    [0, 100, 0, 0, 700],  # Rejeitado
    [0, 100, 0, 0, 700],      # Rejeitado
    [1000, 100, 0, 5, 700],     # Rejeitado
    [1000, 100, 0, 5, 700],     # Rejeitado
    [5000, 2500, 1, 0, 2500],       # Rejeitado
    [5000, 2500, 0, 0, 2500],  # Aprovado
    [200, 100, 0, 0, 50],      # Aprovado
    [5000, 1000, 0, 5, 100],     # Aprovado
    [1000, 3000, 0, 3, 100],     # Aprovado
    [5000, 500, 1, 0, 50]       # Aprovado
]

resposta = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] # Respostas(1=aprovado 0=N/aprovado)

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


# Dados de vários candidatos
candidatos = [
    [4000, 1500, 1, 2, 8000],  # Candidato 1
    [7000, 3000, 0, 1, 10000],  # Candidato 2
    [2000, 1000, 1, 4, 5000],   # Candidato 3
    [0, 5000, 0, 0, 1000],   # Candidato 4
    [10000, 9000, 1, 7, 5000],  # Candidato 5
    [0, 100, 0, 0, 700],    # Candidato 6
    [5000, 5000, 1, 10, 100000],    # Candidato 7
    [1000, 500, 1, 0, 50],  # Candidato 8
    [7000, 5000, 0, 2, 2500]    # Candidato 9
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
