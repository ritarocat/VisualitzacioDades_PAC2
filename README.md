# VisualitzacioDades_PAC2
Visualització de dades - PAC 2

## Descripció del Projecte

Aquest projecte implementa tres tècniques de visualització de dades diferents utilitzant Python i Plotly

### 1. Timeline Plot
Mostra l'evolució dels 10 gèneres cinematogràfics més populars des de 1900 fins l'actualitat.

### 2. Correlation Matrix
Analitza les correlacions entre indicadors socioeconòmics del WorldBank per l'any 2023 mitjançant un mapa de calor interactiu.

### 3. Bullet Graph
Compara el rendiment dels gèneres cinematogràfics utilitzant ratings mitjans per gènere, actuals (últimes 50 películes registrades) i el millor any històric com a target.

## Fonts de Dades

### Dades cinematogràfiques (Tècniques 1 i 3)
**Font:** https://datasets.imdbws.com/
- **Primera tècnica:** `title.basics.tsv`
- **Tercera tècnica:** Join de `title.basics.tsv` i `title.ratings.tsv`

### Dades socioeconòmiques (Tècnica 2)
**Font:** https://databank.worldbank.org/source/world-development-indicators#
- Filtrant només algunes columnes i l'any 2023

## Execució

```bash
python VD_PAC2_RitaRocaTaxonera.py
```

El programa genera tres visualitzacions interactives en format HTML:
- `timeline_graph.html`
- `worldbank_2023_correlation_matrix.html`  
- `bullet_graph_genres.html`
