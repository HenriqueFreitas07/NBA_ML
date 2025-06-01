# 🏀 NBA Matchup Win Prediction

Este projeto utiliza modelos de machine learning (incluindo PyTorch e XGBoost) para prever a probabilidade de vitória em jogos da NBA com base em métricas de impacto de jogadores, estatísticas históricas, ELO ratings e vantagem de jogar em casa.

---

## 📘 Introdução

Este projeto foi desenvolvido no âmbito da unidade curricular **Tópicos de Aprendizagem Automática** e tem como objetivo aplicar técnicas de *machine learning* à previsão de resultados de jogos da NBA.

Através da recolha e análise de dados históricos — estatísticas de jogadores, desempenho por equipa, fator casa/fora e ratings ELO — foram construídos modelos que estimam a probabilidade de vitória num determinado confronto entre duas equipas.

A estrutura do projeto foi organizada para facilitar o processo de **análise de dados**, **engenharia de features**, **treino de modelos supervisionados** e **avaliação de desempenho preditivo**, com enfoque na interpretação dos resultados. Foram utilizados modelos baseados em redes neuronais (*PyTorch*) e algoritmos como o *XGBoost*, com comparações entre abordagens uni e bidireccionais para aumentar a robustez das previsões.

O projeto demonstra não só a aplicação prática de conceitos abordados na disciplina, como também o potencial da aprendizagem automática em contextos reais de desporto e análise preditiva.

## 📁 Estrutura do Projeto
O projecto utiliza maioritariamente um dataset de todas os dados de cada jogador de cada jogo em cada equipa desde 2010 até à season de 2024.
O projecto é constituído por 2 modelos de Machine Learning.
- **XGBoost tunned** -> para o cálculo do impacto de um jogador de uma equipa de uma season podendo este encontrar-se em mais que um registo durante um matchup
- **NN PyTorch** -> para o cálculo das probabilidades entre 2 equipas de NBA, tendo em conta o local do jogo e a season em que o modelo foi treinado
### Data Exploration
A exploração dos dados está presente no jupyterNotebook: ```dataExploration.ipynb```

## PlayerImpact Model
Para a realização do primeiro ficheiro estão presentes os seguintes jupyterNotebook: 
- ```tuning_experiments.ipynb```
- ```featureSelection.ipynb```
- ```dataAggregation.ipynb```
- ```elo_rating.ipynb```

## Matchup Probability Prediction
Para o modelo de previsão da probabilidade de vitória os seguintes contêm o código associado à execução e treino dos mesmos
- ```mode.py```
- ```TrainModel.ipynb```
- ```ModelPipeline.ipynb```
---
# 🏀 Making Predictions
The `ModelPipeline.py` script predicts win probabilities between two NBA teams using player impact values, team ELO ratings, and home/away game context. The script loads a trained model and performs a forward and reversed matchup prediction, averaging both to improve accuracy.

## ⚠️ Notas sobre o Modelo Treinado

Se ocorrer um erro durante a previsão — como dados em falta ou incompatibilidades com a arquitetura do modelo — será apresentada uma mensagem de erro detalhada no `stderr`.

### ❗ Importante

O modelo treinado depende diretamente da época (season) utilizada no seu treino. Isto significa que:

- O **número mínimo de jogadores por equipa** considerado durante o treino pode variar de época para época.
- Se utilizares o modelo com dados de uma época diferente daquela em que foi treinado, ele poderá ainda assim produzir resultados razoáveis, **desde que** a estrutura e disponibilidade de dados das equipas seja semelhante.
- No entanto, o uso com épocas diferentes pode também:
  - Causar **erros em tempo de execução** (por exemplo, se houver menos jogadores disponíveis do que o número esperado).
  - Reduzir significativamente a **precisão das previsões**.

### ✅ Recomendação

Para garantir melhores resultados:
- Treina o modelo com os dados da **mesma época** que vais usar para as previsões.
- Verifica que todas as equipas têm o número mínimo de jogadores necessários.
## ⚙️ Usage

```bash
./ModelPipeline.py --season SEASON_YEAR --matchup TEAM1 TEAM2 [--home]
```