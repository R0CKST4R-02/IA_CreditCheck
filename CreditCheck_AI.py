# Primeira parte do codido dados
from sklearn.model_selection import train_test_split
# Dados fictícios: [salário, depósito mensal, histórico (0=limpo, 1=atrasado), dependentes, valor solicitado]
dados = [
    [5000, 2000, 0, 2, 10000],  # Aprovado
    [3000, 1500, 1, 3, 20000],  # Rejeitado
    [8000, 3000, 0, 1, 5000],   # Aprovado
    [2500, 1000, 1, 2, 15000],  # Rejeitado
    [6000, 2500, 0, 0, 20000],  # Aprovado
]

resposta = [1, 0, 1, 0, 1]# Respostas(1=aprovado 0=N/aprovado)

# Divisao em terino e teste (80%/20%)
x_treino, x_teste, y_treino, y_teste = train_test_split(dados, resposta, test_size=0.2, random_state=42)

#verificar os dados
print("Dados de treino:", x_treino)
print("Dados de treino:", x_teste)
print("Dados de treino:", y_treino)
print("Dados de treino:", y_teste)