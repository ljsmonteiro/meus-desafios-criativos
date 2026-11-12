# 🛡️ Detecção de Fraudes em Transações de Cartão de Crédito

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-green.svg)](https://shap.readthedocs.io/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Projeto prático desenvolvido para o desafio de Machine Learning da **DIO (Digital Innovation One)**, focado na detecção de fraudes em cartões de crédito utilizando dados reais de transações financeiras sob extremo desbalanceamento de classes.

---

## 📌 Sumário Executivo

1. [O Problema e o Paradoxo da Acurácia](#-o-problema-e-o-paradoxo-da-acurácia)
2. [Dataset Utilizado](#-dataset-utilizado)
3. [Preparação e Engenharia de Features](#-preparação-e-engenharia-de-features)
4. [Modelagem e Resultados Comparativos](#-modelagem-e-resultados-comparativos)
5. [Ajuste de Limiar (Threshold Tuning)](#-ajuste-de-limiar-threshold-tuning)
6. [Explicabilidade com SHAP](#-explicabilidade-com-shap)
7. [Diferenciais em Relação ao Baseline](#-diferenciais-em-relação-ao-baseline)
8. [Como Executar o Projeto](#-como-executar-o-projeto)
9. [Estrutura do Repositório](#-estrutura-do-repositório)

---

## ⚠️ O Problema e o Paradoxo da Acurácia

No cenário real de prevenção a fraudes em transações financeiras:
- **99,83%** das transações são legítimas (classe `0`).
- Apenas **0,17%** das transações correspondem a fraudes (classe `1`).

### O "Paradoxo da Acurácia" (*Accuracy Paradox*)
Se construirmos um modelo ingênuo que simplesmente responda *"não é fraude"* para toda e qualquer transação, esse modelo atingirá **99,83% de acurácia global**, mas **zero** fraudes serão detectadas (Recall = 0.0%). Em um banco ou fintech, isso geraria perdas financeiras catastróficas.

Por essa razão, o foco central deste projeto é a avaliação rigorosa das métricas voltadas à classe minoritária:
- **Recall (Sensibilidade):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Percentual de fraudes reais capturadas.
- **Precisão:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Percentual de fraudes confirmadas dentre todas que o modelo alertou.
- **F1-Score:** Média harmônica entre Precisão e Recall.
- **PR-AUC (Average Precision):** Área sob a curva Precisão-Recall, métrica muito mais fidedigna e sensível que a tradicional ROC-AUC em cenários de forte desbalanceamento.

---

## 📂 Dataset Utilizado

Os dados são oriundos de transações europeias de cartões de crédito realizadas em setembro de 2013:
- **Total de registros:** 284.807 transações.
- **Fraudes:** 492 transações (0,172%).
- **Features anônimas:** $V_1$ a $V_{28}$ resultantes de transformação PCA (Principal Component Analysis) para proteção de dados e privacidade.
- **Features originais:**
  - `Time`: segundos decorridos entre a transação e a primeira transação da base.
  - `Amount`: valor financeiro da transação.
  - `Class`: rótulo alvo (0 = normal, 1 = fraude).

> ⚠️ **Atenção:** Em conformidade com as boas práticas e limites do GitHub (<100MB), a base de dados **não** é versionada neste repositório. O notebook consome o arquivo diretamente por streaming via link público do Google/TensorFlow:  
> `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

---

## 🛠️ Preparação e Engenharia de Features

1. **Transformação Logarítmica (`np.log1p`):**  
   A coluna `Amount` possui distribuição com forte assimetria à direita (*long-tail*). A transformação $\log(1 + x)$ comprime a variância e aproxima os valores de uma distribuição mais simétrica.
2. **Padronização (`StandardScaler`):**  
   As features `Time` e `Amount_log` foram normalizadas para média zero e desvio padrão unitário, alinhando-se à escala dos componentes principais ($V_1$ a $V_{28}$).
3. **Divisão Estratificada (`stratify=y`):**  
   Divisão em **80% treino** e **20% teste** mantendo exatamente a proporção de 0,17% de fraudes em ambos os cortes (394 fraudes no treino e 98 fraudes no teste).

---

## 🤖 Modelagem e Resultados Comparativos

Comparamos quatro algoritmos representativos, variando o tratamento de pesos e arquitetura:

| Modelo | Recall (Fraude) | Precisão (Fraude) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | 0.6224 | **0.8714** | 0.7262 | 0.7634 | 0.9712 |
| **Regressão Logística (Balanced)** | **0.9184** | 0.0583 | 0.1097 | 0.7321 | 0.9765 |
| **Random Forest (`class_weight='balanced'`)** | 0.8163 | 0.8696 | **0.8421** | **0.8584** | **0.9821** |
| **XGBoost Classifier (`scale_pos_weight`)** | 0.8469 | 0.8300 | 0.8384 | 0.8512 | 0.9798 |

### Diagnóstico dos Modelos:
- **Regressão Logística Baseline:** Deixou passar quase 38% das fraudes reais (recall baixo).
- **Regressão Logística Balanceada:** Elevou o recall para ~92%, porém gerou elevado volume de falsos positivos (precisão de ~6%).
- **Random Forest & XGBoost:** Entregaram o melhor equilíbrio operacional, superando 80-84% de recall mantendo precisão acima de 83% e **PR-AUC superior a 0.85**.

---

## 🎯 Ajuste de Limiar (Threshold Tuning)

No mercado de pagamentos, **o custo de um Falso Negativo (fraude não detectada que vira chargeback) é muito maior que o de um Falso Positivo (uma validação preventiva via SMS ou token)**.

No notebook, inserimos o modelo em um `Pipeline` e avaliamos a variação do limiar de classificação com o Random Forest:

| Limiar (*Threshold*) | Recall (Fraude) | Fraudes Detectadas | Precisão | F1-Score | Falsos Positivos |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50 (Padrão)** | 81.63% | 80 / 98 | 86.96% | 0.8421 | 12 |
| **0.40** | 82.65% | 81 / 98 | 83.51% | 0.8308 | 16 |
| **0.30 (Recomendado)** | **84.69%** | **83 / 98** | **80.58%** | **0.8259** | **20** |
| **0.20** | 86.73% | 85 / 98 | 73.28% | 0.7944 | 31 |
| **0.10** | 88.78% | 87 / 98 | 58.78% | 0.7073 | 61 |

> 💡 **Conclusão de Negócio:** A redução do limiar para **0.30** permitiu interceptar **3 fraudes adicionais no teste** gerando apenas 8 alertas falsos adicionais em um universo de mais de 56 mil transações.

---

## 🔍 Explicabilidade com SHAP

A explicabilidade é mandatória em serviços financeiros para compliance com LGPD e auditoria de modelos.

Utilizamos o **SHAP (SHapley Additive exPlanations)** com `TreeExplainer` no modelo de ensemble:
- **Variáveis mais determinantes:** $V_{14}$, $V_{10}$, $V_{12}$, $V_4$ e $V_{11}$.
- **Comportamento Direcional:**
  - Valores fortemente negativos de $V_{14}$ e $V_{10}$ aumentam drasticamente a probabilidade de fraude.
  - Valores elevados de $V_4$ atuam como forte gatilho para elevar o risco de fraude.
  - O valor transacionado (`Scaled_Amount`), quando associado a desvios específicos em $V_{14}$ e $V_{12}$, amplifica o escore de risco.

---

## 💡 Diferenciais em Relação ao Baseline / Aulas

1. **Curva Precision-Recall Dedicada:** Implementação gráfica completa comparando ROC-AUC versus PR-AUC, demonstrando o efeito do desbalanceamento na avaliação visual.
2. **Pipelines Robustos:** Junção de pré-processamento e estimadores dentro do `sklearn.pipeline.Pipeline`, prevenindo *Data Leakage*.
3. **Simulação de Estratégias de Reamostragem:** Comparação direta no notebook entre **Random Undersampling** e **Oversampling com SMOTE**, demonstrando o impacto prático em falsos positivos vs recall.
4. **Explicabilidade Prática com SHAP:** Criação do gráfico `summary_plot` detalhando a direção e impacto de cada feature nas transações fraudulentas.

---

## 💻 Como Executar o Projeto

### Opção 1: Via Google Colab (Recomendado)
1. Faça o upload do arquivo [`deteccao_fraude_cartao.ipynb`](./deteccao_fraude_cartao.ipynb) no [Google Colab](https://colab.research.google.com/).
2. Vá em **Ambiente de execução** > **Executar tudo**.
3. O notebook instalará as dependências ausentes automaticamente e baixará os dados em streaming.

### Opção 2: Localmente via Terminal
```bash
# 1. Clone o repositório
git clone https://github.com/ljsmonteiro/meus-desafios-criativos.git
cd meus-desafios-criativos/deteccao-fraudes-cartao

# 2. Crie e ative um ambiente virtual
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
# source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o Jupyter Notebook
jupyter notebook deteccao_fraude_cartao.ipynb
```

---

## 📁 Estrutura do Repositório

```text
deteccao-fraudes-cartao/
├── deteccao_fraude_cartao.ipynb  # Notebook interativo completo
├── requirements.txt              # Dependências do projeto (scikit-learn, xgboost, shap, etc.)
├── .gitignore                    # Regras para ignorar dados e artefatos pesados
└── README.md                     # Documentação técnica e relatório de resultados
```

---

## 👨‍💻 Autor
Desenvolvido por **Lucas J. S. Monteiro** no âmbito dos Desafios de Inteligência Artificial e Machine Learning da **DIO**.
