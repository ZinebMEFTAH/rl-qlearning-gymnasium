# TP4 - Apprentissage par Renforcement
## Rapport de test Q-learning

### Binôme
- BELLAGH Fatma
- MEZAH Zineb

---

## Résultats des tests Q-learning

### 1. Applications testées

#### Frozen Lake 4x4
- **Nombre d'itérations** : 1 000 épisodes
- **Résultat** : Agent apprend à atteindre le but

#### Frozen Lake 8x8  
- **Nombre d'itérations** : Testé jusqu'à 50 000 épisodes
- **Résultat** : Agent n'atteint pas le but (courbe plate à 0)

#### Taxi
- **Nombre d'itérations** : 20 000 épisodes
- **Résultat** : Q-values augmentent mais bug d'affichage

### 2. Application Cart Pole

#### Modifications apportées
- **Fichier `trainAI.py`** : Ajout de `discretizeState(state)` et `discretizeState(next_state)`
- **Fichier `play.py`** : Ajout de `discretizeState(state)`

#### Résultat
- **Nombre d'itérations** : 30 000 épisodes
- **Performance** : Agent équilibre le poteau

---

## Fichiers modifiés
- `agent.py` - Implémentation complète du Q-learning
- `trainAI.py` - Ajout discrétisation Cart Pole
- `play.py` - Ajout discrétisation Cart Pole