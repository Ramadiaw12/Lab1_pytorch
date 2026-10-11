# Lab1_pytorch
# PyTorch Fundamentals — Gradients & Model Training

## 📌 Présentation

Ce mini-laboratoire a pour objectif de découvrir les fondamentaux de **PyTorch** à travers la mise en œuvre du calcul automatique des gradients et de l'entraînement d'un modèle simple.

L'objectif est de comprendre comment un modèle ajuste ses paramètres pour réduire sa fonction de perte (*loss*) au cours des itérations d'entraînement.

## 🎯 Objectifs pédagogiques

- Comprendre les tenseurs (`Tensor`) et leur rôle dans PyTorch.
- Découvrir le calcul automatique des dérivées avec `torch.autograd`.
- Calculer les gradients des paramètres du modèle.
- Comprendre la différence entre `loss.backward()` et `torch.autograd.grad()`.
- Implémenter une boucle d'entraînement (*training loop*).
- Stocker les valeurs de perte à chaque époque (*epoch*).
- Observer l'évolution de la fonction de perte pendant l'entraînement.
- Comprendre le principe de l'optimisation des paramètres d'un modèle.

## 🛠️ Technologies utilisées

- **Python 3.12**
- **PyTorch** : calcul tensoriel, différentiation automatique et entraînement.
- **NumPy** : manipulation de tableaux numériques.
- **Matplotlib** : visualisation des données et de la perte.
- **uv** : gestion de l'environnement Python et des dépendances.
- **Jupyter Notebook** : expérimentation interactive.

## 📚 Notions étudiées

### 1. Tenseurs et paramètres du modèle

Les tenseurs constituent la structure de données fondamentale de PyTorch. Les paramètres d'un modèle peuvent être représentés par des tenseurs dont les gradients sont suivis grâce à `requires_grad=True`.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
```

- `w` : poids (*weight*) du modèle.
- `b` : biais (*bias*) du modèle.
- `requires_grad=True` : demande à PyTorch de suivre les opérations réalisées sur ces paramètres afin de calculer leurs gradients.

### 2. Calcul automatique des gradients avec `torch.autograd`

Le gradient mesure la variation de la fonction de perte par rapport à un paramètre. Il permet de déterminer dans quelle direction ajuster les paramètres pour réduire la perte.

#### Méthode A — `torch.autograd.grad()`

```python
dw, db = torch.autograd.grad(loss, (w, b))
```

Cette instruction calcule les dérivées de `loss` par rapport à `w` et `b`.

- `dw` : gradient de la perte par rapport au poids.
- `db` : gradient de la perte par rapport au biais.

Cette méthode retourne directement les gradients dans des variables Python, sans les accumuler dans les attributs `.grad` des paramètres.

#### Méthode B - `loss.backward()`

```python
loss.backward()

print(w.grad)
print(b.grad)
```

La méthode `backward()` calcule les gradients et les stocke dans les attributs `.grad` des tenseurs concernés.

**Différence essentielle :** `torch.autograd.grad()` retourne les gradients, tandis que `loss.backward()` les accumule dans les attributs `.grad`.

> **Attention :** les gradients stockés par `backward()` s'accumulent entre plusieurs appels. Dans une boucle d'entraînement, il faut généralement les réinitialiser avant le calcul suivant, par exemple avec `optimizer.zero_grad()`.

### 3. Entraînement du modèle (*Training the Model*)

L'entraînement consiste à ajuster les paramètres du modèle pour minimiser sa fonction de perte.

Une boucle d'entraînement classique comprend les étapes suivantes :

1. **Forward pass** : calculer les prédictions du modèle.
2. **Loss computation** : calculer l'écart entre les prédictions et les valeurs attendues.
3. **Backward pass** : calculer les gradients avec `loss.backward()`.
4. **Optimization step** : mettre à jour les paramètres à l'aide d'un optimiseur.
5. **Loss tracking** : enregistrer la valeur de perte pour suivre la progression de l'entraînement.

Exemple de boucle d'entraînement :

```python
loss_history = []

for epoch in range(num_epochs):
    # Réinitialiser les gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x)

    # Calcul de la perte
    loss = criterion(y_pred, y)

    # Backward pass
    loss.backward()

    # Mise à jour des paramètres
    optimizer.step()

    # Stocker la perte de cette époque
    loss_history.append(loss.item())
```

Dans cet exemple :

- `optimizer.zero_grad()` remet les gradients à zéro.
- `loss.backward()` calcule les gradients.
- `optimizer.step()` ajuste les paramètres du modèle.
- `loss.item()` extrait la valeur numérique de la perte.
- `loss_history` conserve une valeur de perte par époque.

*Remarque : cet exemple suppose que `model`, `criterion`, `optimizer`, `x`, `y` et `num_epochs` ont déjà été définis.*

### 4. Stockage des pertes à chaque époque

Pour suivre l'évolution de l'entraînement, une liste Python permet d'enregistrer les valeurs successives de la fonction de perte.

```python
loss_history = []

# À la fin de chaque époque
loss_history.append(loss.item())
```

Cette liste permet notamment de :

- suivre la diminution de la perte ;
- comparer différentes configurations d'entraînement ;
- détecter une stagnation de l'apprentissage ;
- visualiser la convergence du modèle.

### 5. Visualisation de la perte

Matplotlib permet de tracer la courbe d'évolution de la fonction de perte :

```python
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
```

Une diminution générale de la perte indique que le modèle apprend à mieux ajuster ses prédictions aux données d'entraînement. Elle ne garantit toutefois pas, à elle seule, de bonnes performances sur de nouvelles données.

## 🔍 Points clés à retenir

| Notion | Rôle |
|---|---|
| `Tensor` | Représente les données et les paramètres numériques. |
| `requires_grad=True` | Active le suivi des opérations pour le calcul des gradients. |
| `torch.autograd.grad()` | Retourne directement les gradients demandés. |
| `loss.backward()` | Calcule et accumule les gradients dans `.grad`. |
| `optimizer.zero_grad()` | Réinitialise les gradients avant un nouveau calcul. |
| `optimizer.step()` | Met à jour les paramètres du modèle. |
| `loss.item()` | Extrait la valeur numérique de la perte. |
| `loss_history.append()` | Enregistre la perte de chaque époque. |

## 🚀 Installation et exécution


### 1. Cloner le dépôt

```bash
git clone https://github.com/Ramadiaw12/Lab1_pytorch.git
cd Lab1_pytorch
```

### 2. Installer les dépendances

Si le projet contient un fichier `pyproject.toml` et son fichier de verrouillage `uv.lock` :

```bash
uv sync
```

### 3. Lancer le notebook

```bash
uv run jupyter notebook
```

Si Jupyter n'est pas encore installé dans l'environnement :

```bash
uv add --dev jupyter ipykernel
```

Puis relancer la commande de lancement.


## 👨‍💻 Auteur

**DIAWANE Ramatoulaye**  

---

*Mini-lab d'apprentissage - PyTorch, calcul automatique des gradients et entraînement de modèles.*
