from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Dados fictícios: [salário, depósito mensal, histórico (0=limpo, 1=atrasado), dependentes, valor solicitado]
dados = [
    [5000, 2000, 0, 2, 10000],  # Aprovado
    [3000, 1500, 1, 3, 20000],  # Rejeitado
    [8000, 3000, 0, 1, 5000],   # Aprovado
    [2500, 1000, 1, 2, 15000],  # Rejeitado
    [6000, 2500, 0, 0, 20000],  # Aprovado
    [10000, 5000, 1, 4, 2500],  # Aprovado
    [1000, 0, 0, 2, 5000],      # Rejeitado
    [0, 5000, 0, 0, 10000],     # Rejeitado
    [1000, 200, 0, 0, 9000]     # Rejeitado
]

resposta = [1, 0, 1, 0, 1, 1, 0, 0, 0] # Respostas(1=aprovado 0=N/aprovado)

# Divisao em tereino e teste (80%/20%)
x_treino, x_teste, y_treino, y_teste = train_test_split(dados, resposta, test_size=0.2, random_state=42)

# modelo de treino
modelo = DecisionTreeClassifier(max_depth=3, random_state=42) # max_depth=3 -> numero de pergunats que ela faz
modelo.fit(x_treino, y_treino)

# fazer previsoes
previsoes = modelo.predict(x_teste)
# Calcular a precisão
precisao = accuracy_score(y_teste, previsoes)

print("Precisao: ", precisao)
print("Previsões:", previsoes)
print("Valores reais:", y_teste)

plt.figure(figsize=(12, 8))
plot_tree(modelo, feature_names=["Salário", "Depósito Mensal", "Histórico", "Dependentes", "Valor Solicitado"], class_names=["Rejeitado", "Aprovado"], filled=True)
plt.show()

'''
# Dados do novo candidato
novo_candidato = [[4000, 1500, 1, 2, 8000]]
# Fazer a previsão
resultado = modelo.predict(novo_candidato)
# Exibir o resultado
if resultado[0] == 1:
    print("Aprovado")
else:
    print("Rejeitado")

'''
# Dados de vários candidatos
candidatos = [
    [4000, 1500, 1, 2, 8000],  # Candidato 1
    [7000, 3000, 0, 1, 10000], # Candidato 2
    [2000, 1000, 1, 4, 5000]   # Candidato 3
]

# Fazer previsões
resultados = modelo.predict(candidatos)

# Exibir resultados
for i, resultado in enumerate(resultados):
    print(f"Candidato {i + 1}: {'Aprovado' if resultado == 1 else 'Rejeitado'}")
