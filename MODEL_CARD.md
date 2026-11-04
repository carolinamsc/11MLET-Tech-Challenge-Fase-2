# Model Card: propensão de compra

## Uso pretendido

Estimar a probabilidade de uma sessão de navegação terminar em compra, como exercício acadêmico de Engenharia de Machine Learning (11MLET, Fase 2). O modelo não deve ser usado sozinho para decisões sobre clientes reais.

## Dados

Online Shoppers Purchasing Intention Dataset (UCI Machine Learning Repository): 12.330 sessões, 17 atributos e o alvo `Revenue` (~15,5% de compras). Licença CC BY 4.0; cite a fonte: https://doi.org/10.24432/C5F88Q.

Partições estratificadas com seed 42: 60% treino, 20% validação (seleção de modelo) e 20% teste (avaliação final, nunca usado na seleção).

## Modelos avaliados

Ambos usam o mesmo pré-processamento dentro do `Pipeline` do Scikit-Learn: imputação pela mediana e `StandardScaler` nas numéricas; imputação pela moda e `OneHotEncoder` nas categóricas. Os dois usam `class_weight=balanced`.

| Modelo | Hiperparâmetros | Val. F1 | Val. ROC AUC | Val. Recall | Val. Precision |
| --- | --- | --- | --- | --- | --- |
| Regressão logística | C=1.0, max_iter=1000 | 0.609 | 0.908 | 0.790 | 0.495 |
| **Random Forest** (campeão) | 300 árvores, max_depth=12, min_samples_leaf=2 | **0.691** | **0.930** | 0.785 | 0.618 |

Critério de seleção: maior F1 de validação, por ser o que melhor equilibra precisão e recall na classe minoritária.

## Desempenho no teste (modelo campeão)

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.882 |
| Precision | 0.599 |
| Recall | 0.715 |
| F1 | 0.652 |
| ROC AUC | 0.922 |

2.466 sessões de teste. Fonte: `reports/metrics.json`, também logado no run do MLflow e gravado como tags da versão registrada (`11mlet-purchase-intent`, alias `champion`).

## Limitações

- O dataset é de 2018 e de um único e-commerce; não representa necessariamente outros canais, países ou sazonalidades.
- Classes desbalanceadas: com limiar 0,5, cerca de 4 em cada 10 sessões previstas como compra não compram. O limiar pode ser ajustado conforme o custo de cada erro.
- `PageValues` é a variável mais informativa e depende do tracking analítico do site; mudanças nesse cálculo degradam o modelo.
- Previsões são associações estatísticas, não explicações causais.
