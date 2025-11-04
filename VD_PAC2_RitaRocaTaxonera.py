import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.figure_factory as ff
import numpy as np
from collections import Counter

pio.renderers.default = "browser"

class MoviesVisualizations:
    def __init__(self, dataset_path, ratings_path=None, worldbank_path=None):
        self.dataset_path = dataset_path
        self.ratings_path = ratings_path
        self.worldbank_path = worldbank_path
        self.df = None
        self.ratings_df = None
        self.top10_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror',
                       'Documentary','Romance','Family','Animation','Fantasy']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def dataset_load(self):
        """Carrega el dataset per grafic 1: timeline"""
        print("Carregant el dataset...")
        self.df = pd.read_csv(self.dataset_path, sep='\t')
        print(f"Dataset carregat amb {len(self.df)} entrades")

    def worldbank_dataset_load(self):
        """Carrega el dataset del WorldBank per grafic 2: matriu de correlació"""
        print("Carregant el dataset del WorldBank...")
        self.worldbank_df = pd.read_csv(self.worldbank_path, sep=',')
    
    def ratings_dataset_load(self):
        """Carrega el dataset de ratings per grafic 3: bullet graph"""
        if self.ratings_path:
            print("Carregant el dataset de ratings...")
            self.ratings_df = pd.read_csv(self.ratings_path, sep='\t')
            print(f"Dataset de ratings carregat amb {len(self.ratings_df)} entrades")
        else:
            print("Error: No s'ha especificat el path del dataset de ratings")

    def join_with_ratings(self, genre_movies):
        """Donades les pel·lícules d'un gènere, uneix les dades de pel·lícules amb les puntuacions"""
        if self.ratings_df is None:
            print("Error: Dataset de ratings no carregat")
            return genre_movies
        
        # Fer el join per la columna tconst
        movies_with_ratings = genre_movies.merge(
            self.ratings_df, 
            on='tconst', 
            how='inner'
        )
        
        print(f"  Després del join amb ratings: {len(movies_with_ratings)} pel·lícules amb puntuacions")
        return movies_with_ratings
    
    def filter_data_for_genre(self, genre):
        """Filtra el dataset per un gènere específic"""
        genre_mask = self.df['genres'].str.contains(genre, case=False, na=False)
        genre_movies = self.df[genre_mask].copy()
        
        # Filtrar només per pel·lícules i anys vàlids
        genre_movies = genre_movies[genre_movies['titleType'] == 'movie']
        genre_movies = genre_movies[genre_movies['startYear'] != '\\N']
        
        # Convertir startYear a numèric i filtrar anys invàlids
        genre_movies['startYear'] = pd.to_numeric(genre_movies['startYear'], errors='coerce')
        genre_movies = genre_movies.dropna(subset=['startYear'])
        genre_movies = genre_movies[genre_movies['startYear'] >= 1900]
        genre_movies = genre_movies[genre_movies['startYear'] <= 2025]
        
        # Eliminem duplicats
        print(f"  Abans d'eliminar duplicats: {len(genre_movies)} títols")
        genre_movies = genre_movies.drop_duplicates(subset=['primaryTitle', 'startYear'], keep='first')
        
        return genre_movies
    
    def create_timeline_plot(self):
        """Crea un gràfic timeline per als 10 gèneres més comuns"""

        fig = go.Figure()

        for i, genre in enumerate(self.top10_genres):
            genre_movies = self.filter_data_for_genre(genre)
            
            if not genre_movies.empty:
                year_counts = genre_movies['startYear'].value_counts().sort_index()
                
                fig.add_trace(go.Scatter(
                    x=year_counts.index,
                    y=year_counts.values,
                    mode='lines',
                    name=genre,
                    line=dict(width=2, color=self.colors[i % len(self.colors)]),
                    hovertemplate=f'<b>{genre}</b><br>Any: %{{x}}<br>Pel·lícules: %{{y}}<extra></extra>'
                ))

                print(f"  {genre}: {len(genre_movies)} movies ({year_counts.index.min()}-{year_counts.index.max()})")
        
        fig.update_layout(
            title=f'Evolució del gènere de les pel·lícules des de 1900 fins l\'actualitat',
            xaxis_title='Any',
            yaxis_title='Nombre de pel·lícules',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            width=1400,
            height=700
        )
        
        return fig
    
    def prepare_data_for_correlation(self):
        """Prepara les dades del WorldBank per crear la matriu de correlació"""
        print("Preparant dades del WorldBank per la matriu de correlació...")
                
        try:            
            # Identificar columnes numèriques
            numeric_columns = self.worldbank_df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"Columnes numèriques trobades: {numeric_columns}")
            
            # Filtrar les dades numèriques i eliminar NaNs
            correlation_df = self.worldbank_df[numeric_columns]

            print(f"Registres amb dades complertes: {len(correlation_df)}")
            # print(f"Columnes per correlació: {list(correlation_df.columns)}")
            
            return correlation_df
            
        except Exception as e:
            print(f"Error carregant el dataset WorldBank: {e}")
            return pd.DataFrame()  # Retornar DataFrame buit si hi ha error
    
    def create_correlation_matrix(self):
        """Crea una matriu de correlació amb les columnes numèriques del dataset WorldBank"""
        correlation_df = self.prepare_data_for_correlation()
        
        if correlation_df.empty:
            print("No hi ha dades suficients per crear la matriu de correlació")
            return None
        
        # Calcular la matriu de correlació
        corr_matrix = correlation_df.corr(method='pearson', min_periods=1)
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.columns),
            annotation_text=corr_matrix.round(2).values,
            showscale=True,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlació: %{z:.2f}<extra></extra>'
        )
        
        # Ajustar la mida del gràfic segons el nombre de variables
        matrix_width = max(1200, min(2000, len(corr_matrix.columns) * 80))
        matrix_height = max(1000, min(1800, len(corr_matrix.columns) * 70))
        
        fig.update_layout(
            title='Matriu de Correlació - Dataset WorldBank 2023',
            width=matrix_width,
            height=matrix_height,
            xaxis={'tickangle': 45, 'tickfont': {'size': 10}, 'side': 'bottom'},
            yaxis={'tickangle': 0, 'tickfont': {'size': 10}},
            margin=dict(l=100, r=100, t=120, b=150)
        )
        
        return fig
    
    def create_bullet_graph_genres(self):
        """Crea un bullet graph sobre els ratings dels gèneres amb mètriques temporals"""
        
        if self.ratings_df is None:
            print("Error: Dataset de ratings no disponible per al bullet graph")
            return None
        
        genre_data = []
        all_ratings = {}
        
        # Primer, calcular tots els ratings per determinar el target
        for genre in self.top10_genres:
            genre_movies = self.filter_data_for_genre(genre)
            
            if not genre_movies.empty:
                movies_with_ratings = self.join_with_ratings(genre_movies)
                if not movies_with_ratings.empty:
                    # Rating mitjà general
                    avg_rating = movies_with_ratings['averageRating'].mean()
                    all_ratings[genre] = avg_rating
                    
                    # Rating actual (50 pel·lícules més recents disponibles)
                    current_movies = movies_with_ratings.nlargest(50, 'startYear')
                    current_rating = current_movies['averageRating'].mean() if not current_movies.empty else avg_rating
                    
                    # Calcular el millor any per aquest gènere (target més representatiu)
                    yearly_ratings = movies_with_ratings.groupby('startYear')['averageRating'].mean()
                    # Només considerar anys amb almenys 3 pel·lícules per evitar outliers
                    yearly_counts = movies_with_ratings.groupby('startYear').size()
                    valid_years = yearly_counts[yearly_counts >= 3].index
                    if len(valid_years) > 0:
                        valid_yearly_ratings = yearly_ratings[yearly_ratings.index.isin(valid_years)]
                        best_year_rating = valid_yearly_ratings.max()
                        best_year = valid_yearly_ratings.idxmax()
                    else:
                        best_year_rating = avg_rating
                        best_year = movies_with_ratings['startYear'].mode().iloc[0] if len(movies_with_ratings) > 0 else 2000
                    
                    genre_data.append({
                        'genre': genre,
                        'avg_rating': avg_rating,
                        'current_rating': current_rating,
                        'movie_count': len(movies_with_ratings),
                        'current_count': len(current_movies),
                        'earliest_year': movies_with_ratings['startYear'].min(),
                        'latest_year': movies_with_ratings['startYear'].max(),
                        'best_year_rating': best_year_rating,
                        'best_year': int(best_year)
                    })
                    
                    print(f"  {genre}: Mitjà={avg_rating:.2f}, Actual={current_rating:.2f}, Millor any={best_year} ({best_year_rating:.2f})")
        
        if not genre_data:
            print("No hi ha dades de ratings disponibles")
            return None
        
        # Crear bullet graph
        fig = go.Figure()
        
        for i, data in enumerate(genre_data):
            genre = data['genre']
            avg_rating = data['avg_rating']
            current_rating = data['current_rating']
            best_year_rating = data['best_year_rating']
            best_year = data['best_year']
            
            # Barra de fons
            fig.add_trace(go.Bar(
                y=[genre],
                x=[10.0],  # Escala completa de ratings
                orientation='h',
                marker=dict(color='lightgray', opacity=0.1),
                name='Màxim' if i == 0 else '',
                showlegend=True if i == 0 else False,
                legendgroup='scale'
            ))
            
            # Barra principal: Rating mitjà (totes les pel·lícules)
            fig.add_trace(go.Bar(
                y=[genre],
                x=[avg_rating],
                orientation='h',
                marker=dict(color="#6eabd6", opacity=0.7),
                name='Rating Mitjà (totes)' if i == 0 else '',
                showlegend=True if i == 0 else False,
                legendgroup='average',
                hovertemplate=f'<b>{genre} - Mitjà</b><br>Rating: {avg_rating:.2f}<br>Anys: {data["earliest_year"]}-{data["latest_year"]}<br>Pel·lícules: {data["movie_count"]}<extra></extra>'
            ))
            
            # Barra interna: Rating actual (50 pel·lícules més recents)
            fig.add_trace(go.Bar(
                y=[genre],
                x=[current_rating],
                orientation='h',
                marker=dict(color="#120E50", opacity=0.9),  # Color taronja més visible
                width=0.2,  # Barra més prima per anar dins la principal
                name='Rating Actual (50 més recents)' if i == 0 else '',
                showlegend=True if i == 0 else False,
                legendgroup='current'
            ))
            
            # Any del maxim rating (target)
            fig.add_trace(go.Scatter(
                x=[best_year_rating],
                y=[genre],
                mode='markers',
                marker=dict(
                    size=50,
                    color='black',
                    opacity=1,
                    symbol='line-ns-open'
                ),
                name='Target (millor any)' if i == 0 else '',
                showlegend=True if i == 0 else False,
                legendgroup='target',
                hovertemplate=f'<b>{genre} - Target</b><br>Millor any: {best_year}<br>Rating: {best_year_rating:.2f}<extra></extra>'
            ))
        
        # Crear llista ordenada de gèneres (un per Y)
        y_categories = [data['genre'] for data in reversed(genre_data)]
        
        fig.update_layout(
            title='Bullet Graph - Rendiment per Gènere<br>' +
                  '<sub>Actual vs Històric vs Target (millor any)</sub>',
            xaxis_title='Rating (1-10)',
            yaxis_title='Gèneres',
            barmode='overlay',
            height=len(genre_data) * 70 + 200,
            width=1200,
            template='plotly_white',
            xaxis=dict(range=[0, 10.5]),
            yaxis=dict(
                categoryorder='array',
                categoryarray=y_categories
            )
        )
        
        return fig

def main():
    dataset_path = r"C:\Users\rocar\OneDrive - HP Inc\UOC_rita\Q3\Q3_VisualitzacioDades\PAC2\title.basics.tsv"
    ratings_path = r"C:\Users\rocar\OneDrive - HP Inc\UOC_rita\Q3\Q3_VisualitzacioDades\PAC2\title.ratings.tsv"
    worldbank_path = r"C:\Users\rocar\OneDrive - HP Inc\UOC_rita\Q3\Q3_VisualitzacioDades\PAC2\WorldBank_2023.csv"
    
    moviesVisualizations = MoviesVisualizations(dataset_path, ratings_path, worldbank_path)

    # Carregar ambdós datasets
    moviesVisualizations.dataset_load()
    moviesVisualizations.ratings_dataset_load()
    moviesVisualizations.worldbank_dataset_load()

    # TECNICA 1: Timeline
    print("\n===================================")
    print("Creant gràfic de timeline")
    print("===================================")
    timelinegraph = moviesVisualizations.create_timeline_plot()
    timelinegraph.show()
    
    html_path = dataset_path.replace('title.basics.tsv', 'timeline_graph.html')
    timelinegraph.write_html(html_path)
    print(f"Gràfic timeline desat a: {html_path}")
    
    # TECNICA 2: Matriu de correlació
    print("\n===================================")
    print("Creant matriu de correlació")
    print("===================================")
    correlation_graph = moviesVisualizations.create_correlation_matrix()
    if correlation_graph:
        correlation_graph.show()

        correlation_html_path = worldbank_path.replace('WorldBank_2023.csv', 'worldbank_2023_correlation_matrix.html')
        correlation_graph.write_html(correlation_html_path)
        print(f"Matriu de correlació desada a: {correlation_html_path}")
    
    # TECNICA 3: Bullet graph
    print("\n===================================")
    print("Creant bullet graph per gèneres")
    print("===================================")
    bullet_graph = moviesVisualizations.create_bullet_graph_genres()
    bullet_graph.show()
    
    bullet_html_path = dataset_path.replace('title.basics.tsv', 'bullet_graph_genres.html')
    bullet_graph.write_html(bullet_html_path)
    print(f"Bullet graph desat a: {bullet_html_path}")

if __name__ == "__main__":
    main()