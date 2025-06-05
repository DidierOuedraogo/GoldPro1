"""
Application Streamlit pour l'analyse des données aurifères
Version 2.2 - Compatible Streamlit Cloud avec statsmodels
Auteur: Didier Ouedraogo, P.Geo
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import warnings

# Gestion des imports optionnels
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy non disponible - fonctionnalités statistiques limitées")

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn non disponible - Isolation Forest désactivé")

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels non disponible - trendline OLS désactivée")

# Configuration de la page
st.set_page_config(
    page_title="Gold Analysis Pro",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS modernes
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-color: #FFD700;
        --secondary-color: #FFA500;
        --background-dark: #0E1117;
        --card-background: #1E2329;
        --text-primary: #FFFFFF;
        --text-secondary: #B8BCC0;
        --success-color: #00D26A;
        --danger-color: #F85149;
        --info-color: #58A6FF;
    }
    
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1E2329;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.3);
        animation: fadeIn 0.8s ease-out;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1E2329 0%, #2D333B 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(255, 215, 0, 0.2);
        border-color: rgba(255, 215, 0, 0.4);
    }
    
    .stat-card h3 {
        color: var(--primary-color);
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .stat-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1E2329;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2D333B;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(255, 215, 0, 0.2);
        color: var(--text-secondary);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3D434B;
        border-color: rgba(255, 215, 0, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1E2329;
        border: none;
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E2329 0%, #2D333B 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Fonctions utilitaires pour la détection des outliers
class OutlierDetector:
    @staticmethod
    def iqr_method(data, column, multiplier=1.5):
        """Méthode IQR (Interquartile Range)"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    @staticmethod
    def zscore_method(data, column, threshold=3):
        """Méthode Z-Score"""
        if SCIPY_AVAILABLE:
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            data_clean = data.dropna(subset=[column])
            outliers = data_clean[z_scores > threshold]
        else:
            # Calcul manuel du z-score
            data_clean = data.dropna(subset=[column])
            mean = data_clean[column].mean()
            std = data_clean[column].std()
            if std > 0:
                z_scores = np.abs((data_clean[column] - mean) / std)
                outliers = data_clean[z_scores > threshold]
            else:
                outliers = pd.DataFrame()
        return outliers, threshold
    
    @staticmethod
    def isolation_forest(data, columns, contamination=0.1):
        """Isolation Forest pour détection multivariée"""
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import IsolationForest
            data_clean = data[columns].dropna()
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers_pred = iso_forest.fit_predict(data_clean)
            outliers = data_clean[outliers_pred == -1]
            return outliers
        else:
            # Alternative : utiliser IQR sur plusieurs colonnes
            outliers_indices = set()
            for col in columns:
                outliers_col, _, _ = OutlierDetector.iqr_method(data, col)
                outliers_indices.update(outliers_col.index)
            return data.loc[list(outliers_indices)]
    
    @staticmethod
    def percentile_method(data, column, lower_pct=5, upper_pct=95):
        """Méthode des percentiles"""
        lower_bound = data[column].quantile(lower_pct/100)
        upper_bound = data[column].quantile(upper_pct/100)
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

# Fonction pour les tests statistiques
def perform_statistical_tests(data, col1, col2):
    """Effectue des tests statistiques sur les données"""
    results = {}
    
    if SCIPY_AVAILABLE:
        # Test de normalité
        _, p_value_1 = stats.shapiro(data[col1].dropna())
        _, p_value_2 = stats.shapiro(data[col2].dropna())
        results['shapiro_test'] = {
            col1: p_value_1,
            col2: p_value_2
        }
        
        # Corrélations
        pearson_corr, pearson_p = stats.pearsonr(
            data[col1].dropna(),
            data[col2].dropna()
        )
        spearman_corr, spearman_p = stats.spearmanr(
            data[col1].dropna(),
            data[col2].dropna()
        )
        results['correlations'] = {
            'pearson': (pearson_corr, pearson_p),
            'spearman': (spearman_corr, spearman_p)
        }
        
        # Test t apparié
        t_stat, t_p_value = stats.ttest_rel(
            data[col1].dropna(),
            data[col2].dropna()
        )
        results['ttest'] = (t_stat, t_p_value)
    else:
        # Calculs simplifiés sans scipy
        corr = data[[col1, col2]].corr().iloc[0, 1]
        results['correlations'] = {
            'pearson': (corr, None),
            'spearman': (None, None)
        }
        results['shapiro_test'] = {col1: None, col2: None}
        results['ttest'] = (None, None)
    
    return results

# Fonctions d'aide pour l'interface
def colored_header(label, description, color_name="yellow-80"):
    """Version simplifiée de colored_header"""
    st.header(label)
    if description:
        st.caption(description)

def add_vertical_space(num):
    """Ajoute un espace vertical"""
    for _ in range(num):
        st.write("")

# Initialisation de l'état de session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'outliers_removed' not in st.session_state:
    st.session_state.outliers_removed = False
if 'outlier_indices' not in st.session_state:
    st.session_state.outlier_indices = []

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>🏆 Gold Analysis Pro</h1>
        <p>Analyse Avancée des Données Aurifères avec Intelligence Artificielle</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("**Développé par:** Didier Ouedraogo, P.Geo | **Version:** 2.2 | **Date:** 2025")
add_vertical_space(2)

# Sidebar
with st.sidebar:
    colored_header(
        label="Panneau de Contrôle",
        description="Gestion des données et paramètres",
        color_name="yellow-80"
    )
    
    # Upload avec style
    with st.expander("📁 **Chargement des Données**", expanded=True):
        uploaded_file = st.file_uploader(
            "Glissez votre fichier ici",
            type=['csv', 'xlsx', 'xls'],
            help="Formats supportés: CSV, Excel. Le fichier doit contenir les colonnes Fire_Assay, Leach_Well, Lithologie, Niveau_Oxydation"
        )
        
        if uploaded_file is not None:
            with st.spinner('Chargement des données...'):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.data = df
                    st.session_state.processed_data = df.copy()
                    st.success(f"✅ {len(df)} échantillons chargés avec succès!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Lignes", len(df))
                    with col2:
                        st.metric("Colonnes", len(df.columns))
                        
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    if st.session_state.data is not None:
        # Sélection des colonnes
        with st.expander("🎯 **Configuration des Colonnes**", expanded=True):
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            # Détection automatique des colonnes
            default_fa = next((col for col in numeric_cols if 'fire' in col.lower() or 'fa' in col.lower()), numeric_cols[0] if numeric_cols else None)
            default_lw = next((col for col in numeric_cols if 'leach' in col.lower() or 'lw' in col.lower()), numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
            default_litho = next((col for col in categorical_cols if 'litho' in col.lower()), categorical_cols[0] if categorical_cols else None)
            default_oxid = next((col for col in categorical_cols if 'oxid' in col.lower() or 'oxyd' in col.lower()), categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0])
            
            if numeric_cols:
                fire_assay_col = st.selectbox(
                    "Fire Assay", 
                    numeric_cols,
                    index=numeric_cols.index(default_fa) if default_fa in numeric_cols else 0,
                    help="Sélectionnez la colonne contenant les valeurs Fire Assay"
                )
                
                leach_well_col = st.selectbox(
                    "Leach Well", 
                    numeric_cols,
                    index=numeric_cols.index(default_lw) if default_lw in numeric_cols else 0,
                    help="Sélectionnez la colonne contenant les valeurs Leach Well"
                )
            else:
                st.error("Aucune colonne numérique trouvée dans le fichier")
                fire_assay_col = None
                leach_well_col = None
            
            if categorical_cols:
                lithology_col = st.selectbox(
                    "Lithologie", 
                    categorical_cols,
                    index=categorical_cols.index(default_litho) if default_litho in categorical_cols else 0,
                    help="Sélectionnez la colonne contenant les types de lithologie"
                )
                
                oxidation_col = st.selectbox(
                    "Niveau d'Oxydation", 
                    categorical_cols,
                    index=categorical_cols.index(default_oxid) if default_oxid in categorical_cols else 0,
                    help="Sélectionnez la colonne contenant les niveaux d'oxydation"
                )
            else:
                st.warning("Aucune colonne catégorielle trouvée dans le fichier")
                lithology_col = None
                oxidation_col = None
        
        # Gestion des outliers
        with st.expander("🎯 **Détection des Outliers**", expanded=False):
            st.info("🔍 Identifiez et gérez les valeurs aberrantes")
            
            # Liste des méthodes disponibles
            methods = ["IQR (Interquartile Range)", "Percentiles"]
            if SCIPY_AVAILABLE:
                methods.insert(1, "Z-Score")
            if SKLEARN_AVAILABLE:
                methods.append("Isolation Forest")
            
            outlier_method = st.selectbox(
                "Méthode de détection",
                methods,
                help="Choisissez la méthode de détection des outliers"
            )
            
            if fire_assay_col and leach_well_col:
                outlier_column = st.selectbox(
                    "Colonne à analyser",
                    [fire_assay_col, leach_well_col, "Les deux"],
                    help="Sélectionnez la colonne pour la détection des outliers"
                )
            else:
                st.error("Veuillez sélectionner les colonnes Fire Assay et Leach Well")
                outlier_column = None
            
            # Paramètres spécifiques
            if outlier_method == "IQR (Interquartile Range)":
                iqr_multiplier = st.slider(
                    "Multiplicateur IQR",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Plus la valeur est élevée, moins il y aura d'outliers détectés"
                )
            elif outlier_method == "Z-Score":
                z_threshold = st.slider(
                    "Seuil Z-Score",
                    min_value=2.0,
                    max_value=4.0,
                    value=3.0,
                    step=0.1,
                    help="Valeurs avec |Z-Score| > seuil sont considérées comme outliers"
                )
            elif outlier_method == "Percentiles":
                col1, col2 = st.columns(2)
                with col1:
                    lower_pct = st.number_input(
                        "Percentile inf. (%)",
                        min_value=0,
                        max_value=25,
                        value=5,
                        help="Valeurs en dessous de ce percentile"
                    )
                with col2:
                    upper_pct = st.number_input(
                        "Percentile sup. (%)",
                        min_value=75,
                        max_value=100,
                        value=95,
                        help="Valeurs au-dessus de ce percentile"
                    )
            elif outlier_method == "Isolation Forest":
                contamination = st.slider(
                    "Taux de contamination",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    help="Proportion estimée d'outliers dans les données"
                )
            
            if st.button("🔍 Détecter les Outliers", type="primary", use_container_width=True):
                if outlier_column:
                    with st.spinner("Détection en cours..."):
                        detector = OutlierDetector()
                        df_work = st.session_state.processed_data.copy()
                        
                        try:
                            if outlier_column == "Les deux":
                                if outlier_method == "IQR (Interquartile Range)":
                                    outliers_fa, _, _ = detector.iqr_method(df_work, fire_assay_col, iqr_multiplier)
                                    outliers_lw, _, _ = detector.iqr_method(df_work, leach_well_col, iqr_multiplier)
                                    outliers = pd.concat([outliers_fa, outliers_lw]).drop_duplicates()
                                elif outlier_method == "Z-Score" and SCIPY_AVAILABLE:
                                    outliers_fa, _ = detector.zscore_method(df_work, fire_assay_col, z_threshold)
                                    outliers_lw, _ = detector.zscore_method(df_work, leach_well_col, z_threshold)
                                    outliers = pd.concat([outliers_fa, outliers_lw]).drop_duplicates()
                                elif outlier_method == "Percentiles":
                                    outliers_fa, _, _ = detector.percentile_method(df_work, fire_assay_col, lower_pct, upper_pct)
                                    outliers_lw, _, _ = detector.percentile_method(df_work, leach_well_col, lower_pct, upper_pct)
                                    outliers = pd.concat([outliers_fa, outliers_lw]).drop_duplicates()
                                elif outlier_method == "Isolation Forest" and SKLEARN_AVAILABLE:
                                    outliers = detector.isolation_forest(df_work, [fire_assay_col, leach_well_col], contamination)
                            else:
                                col_to_analyze = fire_assay_col if outlier_column == fire_assay_col else leach_well_col
                                
                                if outlier_method == "IQR (Interquartile Range)":
                                    outliers, _, _ = detector.iqr_method(df_work, col_to_analyze, iqr_multiplier)
                                elif outlier_method == "Z-Score" and SCIPY_AVAILABLE:
                                    outliers, _ = detector.zscore_method(df_work, col_to_analyze, z_threshold)
                                elif outlier_method == "Percentiles":
                                    outliers, _, _ = detector.percentile_method(df_work, col_to_analyze, lower_pct, upper_pct)
                                elif outlier_method == "Isolation Forest" and SKLEARN_AVAILABLE:
                                    outliers = detector.isolation_forest(df_work, [col_to_analyze], contamination)
                            
                            st.session_state.outlier_indices = outliers.index.tolist()
                            
                            if len(outliers) > 0:
                                st.warning(f"⚠️ {len(outliers)} outliers détectés ({len(outliers)/len(df_work)*100:.1f}%)")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Fire Assay",
                                        f"{outliers[fire_assay_col].mean():.2f} g/t",
                                        f"±{outliers[fire_assay_col].std():.2f}"
                                    )
                                with col2:
                                    st.metric(
                                        "Leach Well",
                                        f"{outliers[leach_well_col].mean():.2f} g/t",
                                        f"±{outliers[leach_well_col].std():.2f}"
                                    )
                                
                                action = st.radio(
                                    "Action à effectuer",
                                    ["Visualiser", "Exclure", "Conserver"],
                                    horizontal=True
                                )
                                
                                if st.button("✅ Appliquer", type="secondary", use_container_width=True):
                                    if action == "Exclure":
                                        st.session_state.processed_data = df_work.drop(outliers.index)
                                        st.session_state.outliers_removed = True
                                        st.success(f"✅ {len(outliers)} outliers exclus")
                                    else:
                                        st.info("✅ Outliers conservés")
                            else:
                                st.success("✅ Aucun outlier détecté avec ces paramètres")
                        except Exception as e:
                            st.error(f"Erreur lors de la détection: {str(e)}")

# Contenu principal
if st.session_state.data is not None:
    df = st.session_state.processed_data
    
    # Vérification des variables
    if 'numeric_cols' not in locals():
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        fire_assay_col = numeric_cols[0] if numeric_cols else None
        leach_well_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None
        lithology_col = categorical_cols[0] if categorical_cols else None
        oxidation_col = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0] if categorical_cols else None
    
    # Indicateurs de performance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <h3>Échantillons Analysés</h3>
                <div class="value">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if fire_assay_col:
            avg_fa = df[fire_assay_col].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Moyenne Fire Assay</h3>
                    <div class="value">{avg_fa:.2f} g/t</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if leach_well_col:
            avg_lw = df[leach_well_col].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Moyenne Leach Well</h3>
                    <div class="value">{avg_lw:.2f} g/t</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if fire_assay_col and leach_well_col:
            correlation = df[[fire_assay_col, leach_well_col]].corr().iloc[0,1]
            st.markdown(f"""
                <div class="stat-card">
                    <h3>Corrélation</h3>
                    <div class="value">{correlation:.3f}</div>
                </div>
            """, unsafe_allow_html=True)
    
    add_vertical_space(2)
    
    # Indicateur outliers
    if st.session_state.outliers_removed:
        st.info(f"ℹ️ Données filtrées: {len(st.session_state.data) - len(df)} outliers exclus")
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Vue d'ensemble",
        "📈 Analyses Statistiques",
        "🔍 Comparaison des Méthodes",
        "🗺️ Analyse par Lithologie",
        "🎯 Gestion des Outliers",
        "💾 Export & Rapports"
    ])
    
    with tab1:
        colored_header(
            label="Vue d'ensemble des Données",
            description="Exploration et filtrage des données",
            color_name="blue-70"
        )
        
        # Filtres
        if lithology_col and oxidation_col:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                selected_litho = st.multiselect(
                    "🗿 Filtrer par lithologie",
                    df[lithology_col].unique(),
                    default=df[lithology_col].unique()
                )
            
            with col2:
                selected_oxid = st.multiselect(
                    "🌡️ Filtrer par niveau d'oxydation",
                    df[oxidation_col].unique(),
                    default=df[oxidation_col].unique()
                )
            
            with col3:
                if st.button("🔄 Réinitialiser", use_container_width=True):
                    st.experimental_rerun()
            
            filtered_df = df[(df[lithology_col].isin(selected_litho)) & 
                            (df[oxidation_col].isin(selected_oxid))]
        else:
            filtered_df = df
            st.warning("Colonnes de lithologie ou d'oxydation non disponibles pour le filtrage")
        
        # Affichage des données
        st.subheader("📋 Tableau des Données")
        
        # Options d'affichage
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            show_stats = st.checkbox("Afficher les statistiques", value=True)
        with col2:
            highlight_outliers = st.checkbox("Surligner les outliers", value=False)
        
        # Affichage du dataframe
        if highlight_outliers and len(st.session_state.outlier_indices) > 0:
            def highlight_outlier_rows(row):
                if row.name in st.session_state.outlier_indices:
                    return ['background-color: rgba(248, 81, 73, 0.2)'] * len(row)
                return [''] * len(row)
            
            styled_df = filtered_df.style.apply(highlight_outlier_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Statistiques descriptives
        if show_stats and fire_assay_col and leach_well_col:
            st.subheader("📊 Statistiques Descriptives")
            stats_df = filtered_df[[fire_assay_col, leach_well_col]].describe()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Fire Assay**")
                for idx, val in stats_df[fire_assay_col].items():
                    st.metric(idx.capitalize(), f"{val:.3f}")
            with col2:
                st.markdown("**Leach Well**")
                for idx, val in stats_df[leach_well_col].items():
                    st.metric(idx.capitalize(), f"{val:.3f}")
    
    with tab2:
        colored_header(
            label="Analyses Statistiques Avancées",
            description="Distributions et tests statistiques",
            color_name="orange-70"
        )
        
        if fire_assay_col and leach_well_col:
            # Options de visualisation
            viz_type = st.radio(
                "Type de visualisation",
                ["Histogrammes", "Box Plots", "Violin Plots", "Distribution cumulée"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if viz_type == "Histogrammes":
                    fig_fa = px.histogram(
                        filtered_df,
                        x=fire_assay_col,
                        title="Distribution Fire Assay",
                        nbins=50,
                        color_discrete_sequence=['#FFD700']
                    )
                    fig_fa.add_vline(
                        x=filtered_df[fire_assay_col].mean(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Moyenne"
                    )
                elif viz_type == "Box Plots":
                    fig_fa = px.box(
                        filtered_df,
                        y=fire_assay_col,
                        x=lithology_col if lithology_col else None,
                        title="Fire Assay par Lithologie" if lithology_col else "Fire Assay",
                        color=lithology_col if lithology_col else None
                    )
                elif viz_type == "Violin Plots":
                    fig_fa = px.violin(
                        filtered_df,
                        y=fire_assay_col,
                        x=lithology_col if lithology_col else None,
                        title="Distribution Fire Assay",
                        color=oxidation_col if oxidation_col else None,
                        box=True
                    )
                else:  # Distribution cumulée
                    sorted_fa = np.sort(filtered_df[fire_assay_col].dropna())
                    cumulative = np.arange(1, len(sorted_fa) + 1) / len(sorted_fa) * 100
                    
                    fig_fa = go.Figure()
                    fig_fa.add_trace(go.Scatter(
                        x=sorted_fa,
                        y=cumulative,
                        mode='lines',
                        name='Fire Assay',
                        line=dict(color='#FFD700', width=3)
                    ))
                    fig_fa.update_layout(
                        title="Distribution Cumulée Fire Assay",
                        xaxis_title="Teneur (g/t)",
                        yaxis_title="Fréquence cumulée (%)"
                    )
                
                st.plotly_chart(fig_fa, use_container_width=True)
            
            with col2:
                if viz_type == "Histogrammes":
                    fig_lw = px.histogram(
                        filtered_df,
                        x=leach_well_col,
                        title="Distribution Leach Well",
                        nbins=50,
                        color_discrete_sequence=['#FFA500']
                    )
                    fig_lw.add_vline(
                        x=filtered_df[leach_well_col].mean(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Moyenne"
                    )
                elif viz_type == "Box Plots":
                    fig_lw = px.box(
                        filtered_df,
                        y=leach_well_col,
                        x=lithology_col if lithology_col else None,
                        title="Leach Well par Lithologie" if lithology_col else "Leach Well",
                        color=lithology_col if lithology_col else None
                    )
                elif viz_type == "Violin Plots":
                    fig_lw = px.violin(
                        filtered_df,
                        y=leach_well_col,
                        x=lithology_col if lithology_col else None,
                        title="Distribution Leach Well",
                        color=oxidation_col if oxidation_col else None,
                        box=True
                    )
                else:  # Distribution cumulée
                    sorted_lw = np.sort(filtered_df[leach_well_col].dropna())
                    cumulative = np.arange(1, len(sorted_lw) + 1) / len(sorted_lw) * 100
                    
                    fig_lw = go.Figure()
                    fig_lw.add_trace(go.Scatter(
                        x=sorted_lw,
                        y=cumulative,
                        mode='lines',
                        name='Leach Well',
                        line=dict(color='#FFA500', width=3)
                    ))
                    fig_lw.update_layout(
                        title="Distribution Cumulée Leach Well",
                        xaxis_title="Teneur (g/t)",
                        yaxis_title="Fréquence cumulée (%)"
                    )
                
                st.plotly_chart(fig_lw, use_container_width=True)
            
            # Tests statistiques
            st.subheader("🧪 Tests Statistiques")
            test_results = perform_statistical_tests(filtered_df, fire_assay_col, leach_well_col)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Test de Normalité (Shapiro-Wilk)**")
                if test_results['shapiro_test'][fire_assay_col] is not None:
                    st.write(f"Fire Assay p-value: {test_results['shapiro_test'][fire_assay_col]:.4f}")
                    st.write(f"Leach Well p-value: {test_results['shapiro_test'][leach_well_col]:.4f}")
                    
                    if test_results['shapiro_test'][fire_assay_col] < 0.05 or test_results['shapiro_test'][leach_well_col] < 0.05:
                        st.warning("⚠️ Distribution non-normale détectée")
                else:
                    st.info("Test non disponible (scipy requis)")
            
            with col2:
                st.markdown("**Tests de Corrélation**")
                pearson_corr, pearson_p = test_results['correlations']['pearson']
                if pearson_corr is not None:
                    st.write(f"Pearson: {pearson_corr:.3f}")
                    if pearson_p is not None:
                        st.write(f"p-value: {pearson_p:.4f}")
                
                spearman_corr, spearman_p = test_results['correlations']['spearman']
                if spearman_corr is not None:
                    st.write(f"Spearman: {spearman_corr:.3f}")
                    if spearman_p is not None:
                        st.write(f"p-value: {spearman_p:.4f}")
            
            with col3:
                st.markdown("**Test t apparié**")
                t_stat, t_p = test_results['ttest']
                if t_stat is not None:
                    st.write(f"Statistique t: {t_stat:.3f}")
                    st.write(f"p-value: {t_p:.4f}")
                    
                    if t_p < 0.05:
                        st.info("✅ Différence significative entre les méthodes")
                else:
                    st.info("Test non disponible (scipy requis)")
    
    with tab3:
        colored_header(
            label="Comparaison Fire Assay vs Leach Well",
            description="Analyse détaillée de la corrélation et des différences",
            color_name="green-70"
        )
        
        if fire_assay_col and leach_well_col:
            # Calculs préliminaires
            filtered_df['Ratio_LW_FA'] = filtered_df[leach_well_col] / filtered_df[fire_assay_col]
            filtered_df['Difference'] = filtered_df[leach_well_col] - filtered_df[fire_assay_col]
            filtered_df['Pct_Difference'] = (filtered_df['Difference'] / filtered_df[fire_assay_col]) * 100
            
            # Graphique principal de corrélation avec gestion de trendline
            if STATSMODELS_AVAILABLE:
                fig_scatter = px.scatter(
                    filtered_df,
                    x=fire_assay_col,
                    y=leach_well_col,
                    color=lithology_col if lithology_col else None,
                    size=fire_assay_col,
                    hover_data=[oxidation_col, 'Ratio_LW_FA', 'Pct_Difference'] if oxidation_col else ['Ratio_LW_FA', 'Pct_Difference'],
                    title="Corrélation Fire Assay vs Leach Well",
                    trendline="ols"
                )
            else:
                fig_scatter = px.scatter(
                    filtered_df,
                    x=fire_assay_col,
                    y=leach_well_col,
                    color=lithology_col if lithology_col else None,
                    size=fire_assay_col,
                    hover_data=[oxidation_col, 'Ratio_LW_FA', 'Pct_Difference'] if oxidation_col else ['Ratio_LW_FA', 'Pct_Difference'],
                    title="Corrélation Fire Assay vs Leach Well"
                )
                
                # Ajout manuel de la ligne de régression
                x = filtered_df[fire_assay_col].values
                y = filtered_df[leach_well_col].values
                
                # Retirer les NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) > 1:
                    # Régression linéaire simple
                    coeffs = np.polyfit(x_clean, y_clean, 1)
                    line_x = np.array([x_clean.min(), x_clean.max()])
                    line_y = coeffs[0] * line_x + coeffs[1]
                    
                    # Calcul du R²
                    y_pred = coeffs[0] * x_clean + coeffs[1]
                    ss_res = np.sum((y_clean - y_pred) ** 2)
                    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode='lines',
                        name=f'Régression (R²={r_squared:.3f})',
                        line=dict(color='blue', width=2)
                    ))
            
            # Ajout de la ligne 1:1
            max_val = max(filtered_df[fire_assay_col].max(), filtered_df[leach_well_col].max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Ligne 1:1',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            ))
            
            # Mise à jour du layout
            fig_scatter.update_layout(
                xaxis_title=f"{fire_assay_col} (g/t)",
                yaxis_title=f"{leach_well_col} (g/t)",
                height=600
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Métriques de performance
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Ratio moyen LW/FA",
                    f"{filtered_df['Ratio_LW_FA'].mean():.2f}",
                    f"σ = {filtered_df['Ratio_LW_FA'].std():.2f}"
                )
            
            with col2:
                st.metric(
                    "Différence moyenne",
                    f"{filtered_df['Difference'].mean():.2f} g/t",
                    f"σ = {filtered_df['Difference'].std():.2f}"
                )
            
            with col3:
                st.metric(
                    "% Différence moyen",
                    f"{filtered_df['Pct_Difference'].mean():.1f}%",
                    f"σ = {filtered_df['Pct_Difference'].std():.1f}%"
                )
            
            with col4:
                rmse = np.sqrt(np.mean(filtered_df['Difference']**2))
                st.metric(
                    "RMSE",
                    f"{rmse:.3f} g/t",
                    help="Root Mean Square Error"
                )
            
            # Analyse par classe de teneur
            st.subheader("📊 Analyse par Classe de Teneur")
            
            # Définition des classes
            col1, col2 = st.columns([3, 1])
            with col1:
                bins_input = st.text_input(
                    "Limites des classes (g/t)",
                    value="0, 0.5, 1, 2, 5, 10, 100",
                    help="Entrez les limites séparées par des virgules"
                )
            
            try:
                bins = [float(x.strip()) for x in bins_input.split(',')]
                labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
                
                filtered_df['Classe_Teneur'] = pd.cut(
                    filtered_df[fire_assay_col],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )
                
                # Statistiques par classe
                class_stats = filtered_df.groupby('Classe_Teneur', observed=True).agg({
                    fire_assay_col: ['count', 'mean'],
                    leach_well_col: 'mean',
                    'Ratio_LW_FA': 'mean',
                    'Pct_Difference': 'mean'
                }).round(2)
                
                # Graphique des ratios par classe
                fig_ratio = px.bar(
                    x=class_stats.index,
                    y=class_stats[('Ratio_LW_FA', 'mean')],
                    title="Ratio LW/FA par Classe de Teneur",
                    labels={'x': 'Classe de Teneur (g/t)', 'y': 'Ratio LW/FA'},
                    color=class_stats[('Ratio_LW_FA', 'mean')],
                    color_continuous_scale='RdYlGn'
                )
                fig_ratio.add_hline(y=1, line_dash="dash", line_color="black", opacity=0.5)
                fig_ratio.update_layout(showlegend=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig_ratio, use_container_width=True)
                with col2:
                    st.dataframe(class_stats, height=300)
                    
            except Exception as e:
                st.error(f"Erreur dans la définition des classes: {e}")
    
    with tab4:
        colored_header(
            label="Analyse par Lithologie et Oxydation",
            description="Comportement des méthodes selon les caractéristiques géologiques",
            color_name="red-70"
        )
        
        if lithology_col and oxidation_col and fire_assay_col and leach_well_col:
            # Tableau croisé dynamique
            st.subheader("📊 Tableau Croisé Dynamique")
            
            # Options pour le tableau croisé
            col1, col2, col3 = st.columns(3)
            with col1:
                pivot_values = st.selectbox(
                    "Valeurs à analyser",
                    ["Moyennes", "Médianes", "Écart-type", "Comptage"],
                    help="Sélectionnez le type d'agrégation"
                )
            with col2:
                pivot_index = st.selectbox(
                    "Lignes",
                    [lithology_col, oxidation_col],
                    help="Variable pour les lignes du tableau"
                )
            with col3:
                pivot_columns = st.selectbox(
                    "Colonnes",
                    [oxidation_col, lithology_col],
                    index=1 if pivot_index == lithology_col else 0,
                    help="Variable pour les colonnes du tableau"
                )
            
            # Création du tableau croisé
            agg_func = {
                "Moyennes": 'mean',
                "Médianes": 'median',
                "Écart-type": 'std',
                "Comptage": 'count'
            }[pivot_values]
            
            pivot_table = pd.pivot_table(
                filtered_df,
                values=[fire_assay_col, leach_well_col],
                index=pivot_index,
                columns=pivot_columns,
                aggfunc=agg_func,
                fill_value=0
            )
            
            # Affichage avec style
            st.dataframe(
                pivot_table.round(2).style.background_gradient(cmap='YlOrRd'),
                use_container_width=True
            )
            
            # Heatmap interactif
            st.subheader("🗺️ Carte de Chaleur des Teneurs")
            
            # Préparation des données pour le heatmap
            heatmap_data = filtered_df.groupby([lithology_col, oxidation_col]).agg({
                fire_assay_col: 'mean',
                leach_well_col: 'mean'
            }).round(2)
            
            if 'Ratio_LW_FA' in filtered_df.columns:
                ratio_data = filtered_df.groupby([lithology_col, oxidation_col])['Ratio_LW_FA'].mean().round(2)
                heatmap_data['Ratio_LW_FA'] = ratio_data
            
            # Sélection de la métrique pour le heatmap
            heatmap_metric = st.selectbox(
                "Métrique à visualiser",
                ["Fire Assay", "Leach Well", "Ratio LW/FA"],
                help="Sélectionnez la métrique pour la carte de chaleur"
            )
            
            metric_map = {
                "Fire Assay": fire_assay_col,
                "Leach Well": leach_well_col,
                "Ratio LW/FA": 'Ratio_LW_FA'
            }
            
            # Création du heatmap
            heatmap_pivot = heatmap_data[metric_map[heatmap_metric]].unstack()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='YlOrRd',
                text=heatmap_pivot.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig_heatmap.update_layout(
                title=f"Heatmap - {heatmap_metric}",
                xaxis_title=pivot_columns,
                yaxis_title=pivot_index,
                height=500
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Analyse de la récupération
            st.subheader("📈 Analyse de la Récupération")
            
            if 'Ratio_LW_FA' in filtered_df.columns:
                recovery_analysis = filtered_df.groupby([lithology_col, oxidation_col]).agg({
                    'Ratio_LW_FA': ['mean', 'std', 'count']
                }).round(2)
                
                recovery_analysis.columns = ['Ratio Moyen', 'Écart-type', 'N Échantillons']
                recovery_analysis['Récupération %'] = recovery_analysis['Ratio Moyen'] * 100
                
                # Graphique 3D
                fig_3d = px.scatter_3d(
                    filtered_df,
                    x=fire_assay_col,
                    y=leach_well_col,
                    z='Ratio_LW_FA',
                    color=lithology_col,
                    size='Ratio_LW_FA',
                    hover_data=[oxidation_col],
                    title="Analyse 3D - Fire Assay vs Leach Well vs Ratio"
                )
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title="Fire Assay (g/t)",
                        yaxis_title="Leach Well (g/t)",
                        zaxis_title="Ratio LW/FA"
                    ),
                    height=600
                )
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig_3d, use_container_width=True)
                with col2:
                    st.dataframe(recovery_analysis, height=400)
        else:
            st.warning("Veuillez sélectionner toutes les colonnes nécessaires pour cette analyse")
    
    with tab5:
        colored_header(
            label="Gestion Avancée des Outliers",
            description="Visualisation et traitement des valeurs aberrantes",
            color_name="violet-70"
        )
        
        if len(st.session_state.outlier_indices) > 0 and fire_assay_col and leach_well_col:
            outliers_df = st.session_state.data.loc[st.session_state.outlier_indices]
            
            # Statistiques des outliers
            st.subheader("📊 Statistiques des Outliers")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Nombre d'outliers",
                    len(outliers_df),
                    f"{len(outliers_df)/len(st.session_state.data)*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Fire Assay moyen",
                    f"{outliers_df[fire_assay_col].mean():.2f} g/t",
                    f"vs {st.session_state.data[fire_assay_col].mean():.2f} global"
                )
            
            with col3:
                st.metric(
                    "Leach Well moyen",
                    f"{outliers_df[leach_well_col].mean():.2f} g/t",
                    f"vs {st.session_state.data[leach_well_col].mean():.2f} global"
                )
            
            with col4:
                outlier_ratio = outliers_df[leach_well_col].mean() / outliers_df[fire_assay_col].mean()
                global_ratio = st.session_state.data[leach_well_col].mean() / st.session_state.data[fire_assay_col].mean()
                st.metric(
                    "Ratio LW/FA",
                    f"{outlier_ratio:.2f}",
                    f"vs {global_ratio:.2f} global"
                )
            
            # Visualisation des outliers
            st.subheader("🎯 Visualisation des Outliers")
            
            # Scatter plot avec mise en évidence
            fig_outliers = go.Figure()
            
            # Points normaux
            normal_indices = [i for i in st.session_state.data.index if i not in st.session_state.outlier_indices]
            normal_df = st.session_state.data.loc[normal_indices]
            
            fig_outliers.add_trace(go.Scatter(
                x=normal_df[fire_assay_col],
                y=normal_df[leach_well_col],
                mode='markers',
                name='Données normales',
                marker=dict(
                    size=8,
                    color='lightblue',
                    opacity=0.6
                )
            ))
            
            # Outliers
            fig_outliers.add_trace(go.Scatter(
                x=outliers_df[fire_assay_col],
                y=outliers_df[leach_well_col],
                mode='markers',
                name='Outliers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"Index: {idx}<br>FA: {row[fire_assay_col]:.2f}<br>LW: {row[leach_well_col]:.2f}" 
                      for idx, row in outliers_df.iterrows()],
                hoverinfo='text'
            ))
            
            # Ligne 1:1
            max_val = max(st.session_state.data[fire_assay_col].max(), 
                         st.session_state.data[leach_well_col].max())
            fig_outliers.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Ligne 1:1',
                line=dict(color='gray', dash='dash')
            ))
            
            fig_outliers.update_layout(
                title="Distribution des Outliers",
                xaxis_title=f"{fire_assay_col} (g/t)",
                yaxis_title=f"{leach_well_col} (g/t)",
                height=600
            )
            
            st.plotly_chart(fig_outliers, use_container_width=True)
            
            # Tableau détaillé
            st.subheader("📋 Détail des Outliers")
            
            # Options d'affichage
            col1, col2 = st.columns([3, 1])
            with col1:
                columns_to_show = st.multiselect(
                    "Colonnes à afficher",
                    st.session_state.data.columns.tolist(),
                    default=[col for col in [fire_assay_col, leach_well_col, lithology_col, oxidation_col] if col]
                )
            with col2:
                sort_by = st.selectbox(
                    "Trier par",
                    [fire_assay_col, leach_well_col],
                    help="Colonne de tri"
                )
            
            # Affichage du tableau
            if columns_to_show:
                outliers_display = outliers_df[columns_to_show].sort_values(by=sort_by, ascending=False)
                st.dataframe(
                    outliers_display.style.highlight_max(axis=0, color='rgba(255, 215, 0, 0.3)'),
                    use_container_width=True,
                    height=400
                )
            
            # Actions sur les outliers
            st.subheader("⚙️ Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Exclure tous les outliers", type="secondary", use_container_width=True):
                    st.session_state.processed_data = st.session_state.data.drop(st.session_state.outlier_indices)
                    st.session_state.outliers_removed = True
                    st.success(f"✅ {len(st.session_state.outlier_indices)} outliers exclus")
                    st.experimental_rerun()
            
            with col2:
                if st.button("♻️ Réintégrer les outliers", type="secondary", use_container_width=True):
                    st.session_state.processed_data = st.session_state.data.copy()
                    st.session_state.outliers_removed = False
                    st.session_state.outlier_indices = []
                    st.success("✅ Outliers réintégrés")
                    st.experimental_rerun()
            
            with col3:
                # Export des outliers
                csv = outliers_df.to_csv(index=False)
                st.download_button(
                    label="📥 Exporter les outliers",
                    data=csv,
                    file_name=f"outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Aucun outlier détecté. Utilisez le panneau de contrôle pour détecter les outliers.")
    
    with tab6:
        colored_header(
            label="Export des Résultats et Rapports",
            description="Génération de rapports professionnels",
            color_name="blue-green-70"
        )
        
        # Options d'export
        st.subheader("📝 Configuration du Rapport")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Titre du rapport",
                value=f"Analyse des Données Aurifères - {datetime.now().strftime('%Y-%m-%d')}",
                help="Titre qui apparaîtra sur le rapport"
            )
            
            include_options = st.multiselect(
                "Éléments à inclure",
                ["Résumé exécutif", "Statistiques descriptives", "Graphiques", 
                 "Analyse des outliers", "Tests statistiques", "Recommandations"],
                default=["Résumé exécutif", "Statistiques descriptives"]
            )
        
        with col2:
            export_format = st.selectbox(
                "Format d'export",
                ["Excel (Multi-feuilles)", "CSV"],
                help="Sélectionnez le format de sortie"
            )
            
            company_name = st.text_input(
                "Organisation",
                value="",
                help="Nom de votre organisation (optionnel)"
            )
        
        # Aperçu du rapport
        if st.button("👁️ Aperçu du Rapport", type="secondary"):
            st.subheader("📄 Aperçu")
            
            if fire_assay_col and leach_well_col:
                preview_text = f"""
                # {report_title}
                
                **Auteur:** Didier Ouedraogo, P.Geo  
                **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                {'**Organisation:** ' + company_name if company_name else ''}
                
                ## Résumé Exécutif
                
                Cette analyse porte sur {len(filtered_df)} échantillons d'or analysés par les méthodes Fire Assay et Leach Well.
                
                ### Principales conclusions:
                
                - **Teneur moyenne Fire Assay:** {filtered_df[fire_assay_col].mean():.2f} g/t (σ = {filtered_df[fire_assay_col].std():.2f})
                - **Teneur moyenne Leach Well:** {filtered_df[leach_well_col].mean():.2f} g/t (σ = {filtered_df[leach_well_col].std():.2f})
                - **Coefficient de corrélation:** {filtered_df[[fire_assay_col, leach_well_col]].corr().iloc[0,1]:.3f}
                """
                
                if 'Ratio_LW_FA' in filtered_df.columns:
                    preview_text += f"- **Ratio moyen LW/FA:** {filtered_df['Ratio_LW_FA'].mean():.2f}\n"
                
                st.markdown(preview_text)
                
                # Tableau récapitulatif
                if lithology_col:
                    summary_by_litho = filtered_df.groupby(lithology_col).agg({
                        fire_assay_col: ['count', 'mean', 'std'],
                        leach_well_col: ['mean', 'std']
                    }).round(2)
                    
                    if 'Ratio_LW_FA' in filtered_df.columns:
                        ratio_summary = filtered_df.groupby(lithology_col)['Ratio_LW_FA'].mean().round(2)
                        summary_by_litho['Ratio_LW_FA'] = ratio_summary
                    
                    st.dataframe(summary_by_litho, use_container_width=True)
        
        # Génération du rapport
        st.subheader("📊 Génération du Rapport Final")
        
        if st.button("🚀 Générer le Rapport", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Préparation des données...")
                progress_bar.progress(20)
                
                if export_format == "Excel (Multi-feuilles)":
                    # Création du fichier Excel
                    output = io.BytesIO()
                    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        workbook = writer.book
                        
                        # Formats
                        header_format = workbook.add_format({
                            'bold': True,
                            'bg_color': '#FFD700',
                            'font_color': '#000000',
                            'border': 1
                        })
                        
                        # Feuille 1: Données
                        status_text.text("Export des données...")
                        progress_bar.progress(40)
                        filtered_df.to_excel(writer, sheet_name='Données', index=False)
                        
                        # Feuille 2: Statistiques
                        if "Statistiques descriptives" in include_options and fire_assay_col and leach_well_col:
                            status_text.text("Calcul des statistiques...")
                            progress_bar.progress(60)
                            
                            stats_df = filtered_df[[fire_assay_col, leach_well_col]].describe()
                            stats_df.to_excel(writer, sheet_name='Statistiques')
                            
                            if lithology_col and oxidation_col:
                                stats_summary = filtered_df.groupby([lithology_col, oxidation_col]).agg({
                                    fire_assay_col: ['count', 'mean', 'std', 'min', 'max'],
                                    leach_well_col: ['mean', 'std', 'min', 'max']
                                }).round(3)
                                
                                if 'Ratio_LW_FA' in filtered_df.columns:
                                    ratio_stats = filtered_df.groupby([lithology_col, oxidation_col]).agg({
                                        'Ratio_LW_FA': ['mean', 'std']
                                    }).round(3)
                                    stats_summary = pd.concat([stats_summary, ratio_stats], axis=1)
                                
                                stats_summary.to_excel(writer, sheet_name='Statistiques Détaillées')
                        
                        # Feuille 3: Outliers
                        if "Analyse des outliers" in include_options and len(st.session_state.outlier_indices) > 0:
                            status_text.text("Export des outliers...")
                            progress_bar.progress(80)
                            
                            outliers_df = st.session_state.data.loc[st.session_state.outlier_indices]
                            outliers_df.to_excel(writer, sheet_name='Outliers', index=False)
                        
                        # Feuille 4: Métadonnées
                        status_text.text("Finalisation...")
                        progress_bar.progress(90)
                        
                        metadata = pd.DataFrame({
                            'Paramètre': [
                                'Titre du rapport',
                                'Date de génération',
                                'Auteur',
                                'Organisation',
                                'Nombre total d\'échantillons',
                                'Nombre d\'échantillons analysés',
                                'Outliers exclus',
                                'Méthodes d\'analyse',
                                'Version de l\'application'
                            ],
                            'Valeur': [
                                report_title,
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Didier Ouedraogo, P.Geo',
                                company_name if company_name else 'N/A',
                                len(st.session_state.data) if st.session_state.data is not None else 0,
                                len(filtered_df),
                                'Oui' if st.session_state.outliers_removed else 'Non',
                                'Fire Assay, Leach Well',
                                '2.2'
                            ]
                        })
                        metadata.to_excel(writer, sheet_name='Métadonnées', index=False)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Rapport généré avec succès!")
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="📥 Télécharger le rapport Excel",
                        data=output.getvalue(),
                        file_name=f"rapport_analyse_or_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "CSV":
                    # Export CSV simple
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger le CSV",
                        data=csv,
                        file_name=f"donnees_or_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    progress_bar.progress(100)
                    status_text.text("✅ Export CSV généré!")
                
                # Message de succès
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération: {str(e)}")
                progress_bar.empty()
                status_text.empty()

else:
    # Page d'accueil si aucune donnée n'est chargée
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #FFD700;'>Bienvenue dans Gold Analysis Pro</h2>
            <p style='font-size: 1.2rem; margin: 20px 0;'>
                L'outil le plus avancé pour l'analyse des données aurifères
            </p>
            <p style='color: #B8BCC0;'>
                👈 Commencez par charger vos données dans le panneau latéral
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Guide rapide
    with st.expander("📚 Guide de Démarrage Rapide"):
        st.markdown("""
        ### 🚀 Comment utiliser Gold Analysis Pro
        
        1. **Chargement des données**
           - Préparez votre fichier CSV ou Excel avec les colonnes requises
           - Glissez-déposez le fichier dans le panneau latéral
        
        2. **Configuration**
           - Sélectionnez les colonnes correspondantes
           - Configurez les paramètres de détection des outliers
        
        3. **Analyse**
           - Explorez les différents onglets d'analyse
           - Utilisez les filtres interactifs
           - Identifiez et gérez les outliers
        
        4. **Export**
           - Générez des rapports professionnels
           - Exportez dans différents formats
        
        ### 📊 Structure des données requise
        
        Votre fichier doit contenir au minimum:
        - **Fire_Assay**: Résultats Fire Assay (numérique)
        - **Leach_Well**: Résultats Leach Well (numérique)
        - **Lithologie**: Type de roche (texte)
        - **Niveau_Oxydation**: État d'oxydation (texte)
        """)
    
    # Exemple de données
    st.subheader("📋 Exemple de Structure de Données")
    
    example_df = pd.DataFrame({
        'Sample_ID': ['ECH-001', 'ECH-002', 'ECH-003', 'ECH-004', 'ECH-005'],
        'Fire_Assay': [1.25, 0.85, 3.45, 0.55, 2.15],
        'Leach_Well': [1.15, 0.92, 2.98, 0.61, 2.05],
        'Lithologie': ['Granite', 'Schiste', 'Granite', 'Schiste', 'Quartzite'],
        'Niveau_Oxydation': ['Oxydé', 'Frais', 'Oxydé', 'Transition', 'Frais'],
        'Profondeur': [15.5, 25.0, 18.3, 45.2, 31.7]
    })
    
    st.dataframe(
        example_df.style.highlight_max(subset=['Fire_Assay', 'Leach_Well'], color='rgba(255, 215, 0, 0.3)'),
        use_container_width=True
    )
    
    # Bouton de téléchargement de l'exemple
    csv_example = example_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger l'exemple",
        data=csv_example,
        file_name="exemple_donnees_or.csv",
        mime="text/csv"
    )

# Footer moderne
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #1E2329; border-radius: 10px;'>
        <p style='color: #FFD700; font-size: 1.1rem; margin: 0;'>
            🏆 Gold Analysis Pro v2.2
        </p>
        <p style='color: #B8BCC0; margin: 5px 0;'>
            Développé par Didier Ouedraogo, P.Geo
        </p>
        <p style='color: #6C757D; font-size: 0.9rem;'>
            © 2025 - Analyse Avancée des Données Aurifères
        </p>
    </div>
""", unsafe_allow_html=True)