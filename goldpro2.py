"""
Application Streamlit pour l'analyse des données aurifères
Version 2.1 - Compatible Streamlit Cloud
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
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import streamlit_extras
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.add_vertical_space import add_vertical_space
    EXTRAS_AVAILABLE = True
except ImportError:
    EXTRAS_AVAILABLE = False
    # Fonctions de remplacement
    def colored_header(label, description, color_name):
        st.header(label)
        st.caption(description)
    
    def add_vertical_space(num):
        for _ in range(num):
            st.write("")

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
        # Corrélation simple
        corr = data[[col1, col2]].corr().iloc[0, 1]
        results['correlations'] = {
            'pearson': (corr, None),
            'spearman': (None, None)
        }
        results['shapiro_test'] = {col1: None, col2: None}
        results['ttest'] = (None, None)
    
    return results

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
        <p>Analyse Avancée des Données Aurifères</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("**Développé par:** Didier Ouedraogo, P.Geo | **Version:** 2.1 | **Date:** 2025")
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
            help="Formats supportés: CSV, Excel"
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
                    st.success(f"✅ {len(df)} échantillons chargés!")
                    
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
            
            # Valeurs par défaut intelligentes
            default_fa = next((col for col in numeric_cols if 'fire' in col.lower() or 'fa' in col.lower()), numeric_cols[0] if numeric_cols else None)
            default_lw = next((col for col in numeric_cols if 'leach' in col.lower() or 'lw' in col.lower()), numeric_cols[1] if len(numeric_cols) > 1 else None)
            default_litho = next((col for col in categorical_cols if 'litho' in col.lower()), categorical_cols[0] if categorical_cols else None)
            default_oxid = next((col for col in categorical_cols if 'oxid' in col.lower() or 'oxyd' in col.lower()), categorical_cols[1] if len(categorical_cols) > 1 else None)
            
            fire_assay_col = st.selectbox("Fire Assay", numeric_cols, index=numeric_cols.index(default_fa) if default_fa else 0)
            leach_well_col = st.selectbox("Leach Well", numeric_cols, index=numeric_cols.index(default_lw) if default_lw else 0)
            lithology_col = st.selectbox("Lithologie", categorical_cols, index=categorical_cols.index(default_litho) if default_litho else 0)
            oxidation_col = st.selectbox("Niveau d'Oxydation", categorical_cols, index=categorical_cols.index(default_oxid) if default_oxid else 0)
        
        # Gestion des outliers
        with st.expander("🎯 **Détection des Outliers**", expanded=False):
            st.info("🔍 Identifiez et gérez les valeurs aberrantes")
            
            # Liste des méthodes disponibles selon les imports
            methods = ["IQR (Interquartile Range)", "Percentiles"]
            if SCIPY_AVAILABLE:
                methods.insert(1, "Z-Score")
            if SKLEARN_AVAILABLE:
                methods.append("Isolation Forest")
            
            outlier_method = st.selectbox("Méthode de détection", methods)
            
            outlier_column = st.selectbox(
                "Colonne à analyser",
                [fire_assay_col, leach_well_col, "Les deux"]
            )
            
            # Paramètres selon la méthode
            if outlier_method == "IQR (Interquartile Range)":
                iqr_multiplier = st.slider("Multiplicateur IQR", 1.0, 3.0, 1.5, 0.1)
            elif outlier_method == "Z-Score":
                z_threshold = st.slider("Seuil Z-Score", 2.0, 4.0, 3.0, 0.1)
            elif outlier_method == "Percentiles":
                col1, col2 = st.columns(2)
                with col1:
                    lower_pct = st.number_input("Percentile inf. (%)", 0, 25, 5)
                with col2:
                    upper_pct = st.number_input("Percentile sup. (%)", 75, 100, 95)
            elif outlier_method == "Isolation Forest":
                contamination = st.slider("Taux de contamination", 0.01, 0.3, 0.1, 0.01)
            
            if st.button("🔍 Détecter les Outliers", type="primary", use_container_width=True):
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
                                st.metric("Fire Assay", f"{outliers[fire_assay_col].mean():.2f} g/t")
                            with col2:
                                st.metric("Leach Well", f"{outliers[leach_well_col].mean():.2f} g/t")
                            
                            action = st.radio("Action", ["Visualiser", "Exclure", "Conserver"], horizontal=True)
                            
                            if st.button("✅ Appliquer", use_container_width=True):
                                if action == "Exclure":
                                    st.session_state.processed_data = df_work.drop(outliers.index)
                                    st.session_state.outliers_removed = True
                                    st.success(f"✅ {len(outliers)} outliers exclus")
                                else:
                                    st.info("✅ Outliers conservés")
                        else:
                            st.success("✅ Aucun outlier détecté")
                    except Exception as e:
                        st.error(f"Erreur lors de la détection: {str(e)}")

# Contenu principal
if st.session_state.data is not None:
    df = st.session_state.processed_data
    
    # Indicateurs en haut
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <h3>Échantillons</h3>
                <div class="value">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <h3>Fire Assay Moy.</h3>
                <div class="value">{df[fire_assay_col].mean():.2f} g/t</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <h3>Leach Well Moy.</h3>
                <div class="value">{df[leach_well_col].mean():.2f} g/t</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        correlation = df[[fire_assay_col, leach_well_col]].corr().iloc[0,1]
        st.markdown(f"""
            <div class="stat-card">
                <h3>Corrélation</h3>
                <div class="value">{correlation:.3f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    add_vertical_space(2)
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "📈 Analyses Statistiques",
        "🔍 Comparaison",
        "🗺️ Lithologie",
        "💾 Export"
    ])
    
    with tab1:
        st.header("Vue d'ensemble des Données")
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            selected_litho = st.multiselect(
                "Filtrer par lithologie",
                df[lithology_col].unique(),
                default=df[lithology_col].unique()
            )
        with col2:
            selected_oxid = st.multiselect(
                "Filtrer par oxydation",
                df[oxidation_col].unique(),
                default=df[oxidation_col].unique()
            )
        
        filtered_df = df[(df[lithology_col].isin(selected_litho)) & 
                        (df[oxidation_col].isin(selected_oxid))]
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Stats descriptives
        st.subheader("Statistiques Descriptives")
        stats_df = filtered_df[[fire_assay_col, leach_well_col]].describe()
        st.dataframe(stats_df.round(3))
    
    with tab2:
        st.header("Analyses Statistiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme Fire Assay
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
                annotation_text="Moyenne"
            )
            st.plotly_chart(fig_fa, use_container_width=True)
        
        with col2:
            # Histogramme Leach Well
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
                annotation_text="Moyenne"
            )
            st.plotly_chart(fig_lw, use_container_width=True)
        
        # Tests statistiques
        if SCIPY_AVAILABLE:
            st.subheader("Tests Statistiques")
            test_results = perform_statistical_tests(filtered_df, fire_assay_col, leach_well_col)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Test de Normalité (Shapiro-Wilk)**")
                if test_results['shapiro_test'][fire_assay_col] is not None:
                    st.write(f"Fire Assay p-value: {test_results['shapiro_test'][fire_assay_col]:.4f}")
                    st.write(f"Leach Well p-value: {test_results['shapiro_test'][leach_well_col]:.4f}")
            
            with col2:
                st.markdown("**Corrélations**")
                pearson_corr, pearson_p = test_results['correlations']['pearson']
                if pearson_corr is not None:
                    st.write(f"Pearson: {pearson_corr:.3f}")
                    if pearson_p is not None:
                        st.write(f"p-value: {pearson_p:.4f}")
            
            with col3:
                st.markdown("**Test t apparié**")
                t_stat, t_p = test_results['ttest']
                if t_stat is not None:
                    st.write(f"Statistique t: {t_stat:.3f}")
                    st.write(f"p-value: {t_p:.4f}")
    
    with tab3:
        st.header("Comparaison Fire Assay vs Leach Well")
        
        # Calculs
        filtered_df['Ratio_LW_FA'] = filtered_df[leach_well_col] / filtered_df[fire_assay_col]
        filtered_df['Difference'] = filtered_df[leach_well_col] - filtered_df[fire_assay_col]
        filtered_df['Pct_Difference'] = (filtered_df['Difference'] / filtered_df[fire_assay_col]) * 100
        
        # Graphique de corrélation
        fig_scatter = px.scatter(
            filtered_df,
            x=fire_assay_col,
            y=leach_well_col,
            color=lithology_col,
            title="Corrélation Fire Assay vs Leach Well",
            trendline="ols"
        )
        
        # Ligne 1:1
        max_val = max(filtered_df[fire_assay_col].max(), filtered_df[leach_well_col].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Ligne 1:1',
            line=dict(color='red', dash='dash')
        ))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ratio moyen LW/FA", f"{filtered_df['Ratio_LW_FA'].mean():.2f}")
        with col2:
            st.metric("Différence moyenne", f"{filtered_df['Difference'].mean():.2f} g/t")
        with col3:
            st.metric("% Différence", f"{filtered_df['Pct_Difference'].mean():.1f}%")
    
    with tab4:
        st.header("Analyse par Lithologie")
        
        # Tableau croisé
        pivot_table = pd.pivot_table(
            filtered_df,
            values=[fire_assay_col, leach_well_col],
            index=lithology_col,
            columns=oxidation_col,
            aggfunc='mean'
        )
        
        st.dataframe(pivot_table.round(2), use_container_width=True)
        
        # Graphique par lithologie
        fig_litho = px.box(
            filtered_df,
            x=lithology_col,
            y=fire_assay_col,
            color=oxidation_col,
            title="Distribution Fire Assay par Lithologie"
        )
        st.plotly_chart(fig_litho, use_container_width=True)
    
    with tab5:
        st.header("Export des Résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Format d'export",
                ["CSV", "Excel"]
            )
        
        with col2:
            include_stats = st.checkbox("Inclure les statistiques", value=True)
        
        if st.button("📥 Générer l'Export", type="primary", use_container_width=True):
            try:
                if export_format == "CSV":
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name=f"analyse_or_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        filtered_df.to_excel(writer, sheet_name='Données', index=False)
                        
                        if include_stats:
                            stats_df.to_excel(writer, sheet_name='Statistiques')
                            
                            # Résumé par lithologie
                            summary = filtered_df.groupby([lithology_col, oxidation_col]).agg({
                                fire_assay_col: ['count', 'mean', 'std'],
                                leach_well_col: ['mean', 'std']
                            }).round(3)
                            summary.to_excel(writer, sheet_name='Résumé')
                    
                    st.download_button(
                        label="Télécharger Excel",
                        data=buffer.getvalue(),
                        file_name=f"analyse_or_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("✅ Export généré avec succès!")
                
            except Exception as e:
                st.error(f"Erreur lors de l'export: {str(e)}")

else:
    # Page d'accueil
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #FFD700;'>Bienvenue dans Gold Analysis Pro</h2>
            <p style='font-size: 1.2rem;'>
                L'outil d'analyse des données aurifères
            </p>
            <p>👈 Chargez vos données dans le panneau latéral</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Exemple de données
    st.subheader("Format des Données Requis")
    
    example_df = pd.DataFrame({
        'Sample_ID': ['ECH-001', 'ECH-002', 'ECH-003'],
        'Fire_Assay': [1.25, 0.85, 3.45],
        'Leach_Well': [1.15, 0.92, 2.98],
        'Lithologie': ['Granite', 'Schiste', 'Granite'],
        'Niveau_Oxydation': ['Oxydé', 'Frais', 'Oxydé']
    })
    
    st.dataframe(example_df)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #FFD700;'>🏆 Gold Analysis Pro v2.1</p>
        <p>Développé par Didier Ouedraogo, P.Geo | © 2025</p>
    </div>
""", unsafe_allow_html=True)