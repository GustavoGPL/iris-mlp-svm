# Comparação de Classificadores: MLP (Rede Neural) vs. SVM (Support Vector Machine) no Dataset Iris

Este repositório contém um notebook, idealmente executado no **Google Colab**, que implementa e compara o desempenho de classificadores Multi-Layer Perceptron (MLP) e Support Vector Machine (SVM) para a tarefa de classificação de espécies de flores no clássico dataset Iris.

## 📊 Dataset

O projeto utiliza o famoso **Iris dataset** da `scikit-learn`, que é ideal para tarefas de classificação multiclasse. O dataset possui 150 amostras, 50 para cada uma das três espécies de Iris (Setosa, Versicolor e Virginica), com quatro atributos para cada amostra:

* Comprimento da Sépala (sepal length)
* Largura da Sépala (sepal width)
* Comprimento da Pétala (petal length)
* Largura da Pétala (petal width)

O notebook inclui uma fase de Análise Exploratória de Dados (EDA), com estatísticas descritivas, `pairplot` para visualização das relações entre as classes e um mapa de calor para análise de correlação.

## 🧠 Modelos Implementados

Dois algoritmos de classificação foram treinados e comparados:

1.  **Multi-Layer Perceptron (MLP) - Rede Neural**
    * Testado com diferentes arquiteturas de camadas ocultas, como `(4, 8, 4)`, `(8, 16, 8)` e `(16, 32, 16)`.
    * Função de Ativação: `relu`.
    * Otimizador: `adam`.
    * Iterações Máximas: `1000`.

2.  **Support Vector Machine (SVM)**
    * Testado com os kernels: **Linear** e **RBF** (Radial Basis Function).

## ⚙️ Pré-processamento e Avaliação

### Pré-processamento
* Os dados são divididos em conjuntos de treino e teste.
* É aplicado o **StandardScaler** para padronizar os atributos (média zero e desvio padrão unitário), garantindo que todos os modelos trabalhem com dados em escalas comparáveis.

### Métricas de Avaliação
O desempenho de cada modelo é avaliado com base nas seguintes métricas, calculadas para os conjuntos de treino e teste:
* Acurácia (`accuracy_score`)
* Precisão (`precision_score`)
* Recall (`recall_score`)
* F1-Score (`f1_score`)
* Matriz de Confusão (`confusion_matrix`)
* Relatório de Classificação (`classification_report`)

Além disso, são comparados o **tempo de inferência** (em segundos por amostra) e a **complexidade** de cada modelo (número de parâmetros para MLP e número de vetores de suporte para SVM).

## 💻 Requisitos

As bibliotecas Python necessárias estão listadas no arquivo `requirements.txt`:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

Se você optar por rodar localmente (fora do Colab), pode instalá-las usando:

```bash
pip install -r requirements.txt
