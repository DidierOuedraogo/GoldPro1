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
        """Méthode Z-Score simplifiée"""
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
    def percentile_method(data, column, lower_pct=5, upper_pct=95):
        """Méthode des percentiles"""
        lower_bound = data[column].quantile(lower_pct/100)
        upper_bound = data[column].quantile(upper_pct/100)
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

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
        
        # Gestion des outliers
        with st.expander("🎯 **Détection des Outliers**", expanded=False):
            st.info("🔍 Identifiez et gérez les valeurs aberrantes")
            
            outlier_method = st.selectbox(
                "Méthode de détection",
                ["IQR (Interquartile Range)", "Z-Score", "Percentiles"],
                help="Choisissez la méthode de détection des outliers"
            )
            
            outlier_column = st.selectbox(
                "Colonne à analyser",
                [fire_assay_col, leach_well_col, "Les deux"],
                help="Sélectionnez la colonne pour la détection des outliers"
            )
            
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
            else:  # Percentiles
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
                            elif outlier_method == "Z-Score":
                                outliers_fa, _ = detector.zscore_method(df_work, fire_assay_col, z_threshold)
                                outliers_lw, _ = detector.zscore_method(df_work, leach_well_col, z_threshold)
                                outliers = pd.concat([outliers_fa, outliers_lw]).drop_duplicates()
                            else:  # Percentiles
                                outliers_fa, _, _ = detector.percentile_method(df_work, fire_assay_col, lower_pct, upper_pct)
                                outliers_lw, _, _ = detector.percentile_method(df_work, leach_well_col, lower_pct, upper_pct)
                                outliers = pd.concat([outliers_fa, outliers_lw]).drop_duplicates()
                        else:
                            col_to_analyze = fire_assay_col if outlier_column == fire_assay_col else leach_well_col
                            
                            if outlier_method == "IQR (Interquartile Range)":
                                outliers, _, _ = detector.iqr_method(df_work, col_to_analyze, iqr_multiplier)
                            elif outlier_method == "Z-Score":
                                outliers, _ = detector.zscore_method(df_work, col_to_analyze, z_threshold)
                            else:  # Percentiles
                                outliers, _, _ = detector.percentile_method(df_work, col_to_analyze, lower_pct, upper_pct)
                        
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "📈 Analyses Statistiques",
        "🔍 Comparaison des Méthodes",
        "🗺️ Analyse par Lithologie",
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
            col1, col2 = st.columns(2)
            
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
            
            filtered_df = df[(df[lithology_col].isin(selected_litho)) & 
                            (df[oxidation_col].isin(selected_oxid))]
        else:
            filtered_df = df
        
        # Affichage des données
        st.subheader("📋 Tableau des Données")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Statistiques descriptives
        if fire_assay_col and leach_well_col:
            st.subheader("📊 Statistiques Descriptives")
            stats_df = filtered_df[[fire_assay_col, leach_well_col]].describe()
            st.dataframe(stats_df.round(3))
    
    with tab2:
        colored_header(
            label="Analyses Statistiques Détaillées",
            description="Distributions et visualisations",
            color_name="orange-70"
        )
        
        if fire_assay_col and leach_well_col:
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
                    line_color="red",
                    annotation_text="Moyenne"
                )
                st.plotly_chart(fig_fa, use_container_width=True)
                
                # Box plot par lithologie
                if lithology_col:
                    fig_box_litho = px.box(
                        filtered_df,
                        x=lithology_col,
                        y=fire_assay_col,
                        title="Fire Assay par Lithologie"
                    )
                    fig_box_litho.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_box_litho, use_container_width=True)
            
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
                    line_color="red",
                    annotation_text="Moyenne"
                )
                st.plotly_chart(fig_lw, use_container_width=True)
                
                # Box plot par oxydation
                if oxidation_col:
                    fig_box_oxid = px.box(
                        filtered_df,
                        x=oxidation_col,
                        y=leach_well_col,
                        title="Leach Well par Niveau d'Oxydation"
                    )
                    fig_box_oxid.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_box_oxid, use_container_width=True)
    
    with tab3:
        colored_header(
            label="Comparaison Fire Assay vs Leach Well",
            description="Analyse de corrélation et différences",
            color_name="green-70"
        )
        
        if fire_assay_col and leach_well_col:
            # Calculs
            filtered_df['Ratio_LW_FA'] = filtered_df[leach_well_col] / filtered_df[fire_assay_col]
            filtered_df['Difference'] = filtered_df[leach_well_col] - filtered_df[fire_assay_col]
            filtered_df['Pct_Difference'] = (filtered_df['Difference'] / filtered_df[fire_assay_col]) * 100
            
            # Graphique de corrélation
            fig_scatter = px.scatter(
                filtered_df,
                x=fire_assay_col,
                y=leach_well_col,
                color=lithology_col if lithology_col else None,
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
                st.metric("% Différence moyen", f"{filtered_df['Pct_Difference'].mean():.1f}%")
            
            # Analyse par classe de teneur
            st.subheader("Comparaison par Classe de Teneur")
            
            bins = [0, 0.5, 1, 2, 5, 10, 100]
            labels = ['<0.5', '0.5-1', '1-2', '2-5', '5-10', '>10']
            filtered_df['Classe_Teneur'] = pd.cut(filtered_df[fire_assay_col], bins=bins, labels=labels)
            
            class_stats = filtered_df.groupby('Classe_Teneur', observed=True).agg({
                fire_assay_col: ['count', 'mean'],
                leach_well_col: 'mean',
                'Ratio_LW_FA': 'mean'
            }).round(2)
            
            st.dataframe(class_stats)
    
    with tab4:
        colored_header(
            label="Analyse par Lithologie et Oxydation",
            description="Comportement selon les caractéristiques géologiques",
            color_name="red-70"
        )
        
        if lithology_col and oxidation_col and fire_assay_col and leach_well_col:
            # Tableau croisé
            pivot_table = pd.pivot_table(
                filtered_df,
                values=[fire_assay_col, leach_well_col],
                index=lithology_col,
                columns=oxidation_col,
                aggfunc='mean'
            )
            
            st.subheader("Teneurs Moyennes par Lithologie et Oxydation")
            st.dataframe(pivot_table.round(2))
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                fig_fa_group = px.bar(
                    filtered_df.groupby([lithology_col])[fire_assay_col].mean().reset_index(),
                    x=lithology_col,
                    y=fire_assay_col,
                    title="Fire Assay Moyen par Lithologie",
                    color_discrete_sequence=['#FFD700']
                )
                st.plotly_chart(fig_fa_group, use_container_width=True)
            
            with col2:
                fig_lw_group = px.bar(
                    filtered_df.groupby([oxidation_col])[leach_well_col].mean().reset_index(),
                    x=oxidation_col,
                    y=leach_well_col,
                    title="Leach Well Moyen par Oxydation",
                    color_discrete_sequence=['#FFA500']
                )
                st.plotly_chart(fig_lw_group, use_container_width=True)
    
    with tab5:
        colored_header(
            label="Export des Résultats",
            description="Génération de rapports et export de données",
            color_name="blue-green-70"
        )
        
        st.subheader("📝 Configuration du Rapport")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Titre du rapport",
                value=f"Analyse des Données Aurifères - {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            export_format = st.selectbox(
                "Format d'export",
                ["Excel", "CSV"]
            )
        
        with col2:
            include_stats = st.checkbox("Inclure les statistiques", value=True)
            include_charts = st.checkbox("Inclure les graphiques", value=False)
        
        if st.button("📥 Générer le Rapport", type="primary", use_container_width=True):
            try:
                if export_format == "CSV":
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger le CSV",
                        data=csv,
                        file_name=f"analyse_or_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Données principales
                        filtered_df.to_excel(writer, sheet_name='Données', index=False)
                        
                        if include_stats and fire_assay_col and leach_well_col:
                            # Statistiques descriptives
                            stats_df.to_excel(writer, sheet_name='Statistiques')
                            
                            # Résumé par lithologie
                            if lithology_col and oxidation_col:
                                summary = filtered_df.groupby([lithology_col, oxidation_col]).agg({
                                    fire_assay_col: ['count', 'mean', 'std'],
                                    leach_well_col: ['mean', 'std']
                                }).round(3)
                                summary.to_excel(writer, sheet_name='Résumé')
                        
                        # Métadonnées
                        metadata = pd.DataFrame({
                            'Paramètre': ['Titre', 'Date', 'Auteur', 'Échantillons analysés', 'Outliers exclus'],
                            'Valeur': [
                                report_title,
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Didier Ouedraogo, P.Geo',
                                len(filtered_df),
                                'Oui' if st.session_state.outliers_removed else 'Non'
                            ]
                        })
                        metadata.to_excel(writer, sheet_name='Métadonnées', index=False)
                    
                    st.download_button(
                        label="Télécharger le Rapport Excel",
                        data=buffer.getvalue(),
                        file_name=f"rapport_analyse_or_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("✅ Rapport généré avec succès!")
                
            except Exception as e:
                st.error(f"Erreur lors de la génération: {str(e)}")

else:
    # Page d'accueil
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #FFD700;'>Bienvenue dans Gold Analysis Pro</h2>
            <p style='font-size: 1.2rem; margin: 20px 0;'>
                L'outil professionnel pour l'analyse des données aurifères
            </p>
            <p style='color: #B8BCC0;'>
                👈 Commencez par charger vos données dans le panneau latéral
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Exemple de structure
    st.subheader("📋 Structure de Données Attendue")
    
    example_data = pd.DataFrame({
        'Sample_ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
        'Fire_Assay': [1.25, 0.85, 3.45, 0.55, 2.15],
        'Leach_Well': [1.15, 0.92, 2.98, 0.61, 2.05],
        'Lithologie': ['Granite', 'Schiste', 'Granite', 'Schiste', 'Quartzite'],
        'Niveau_Oxydation': ['Oxydé', 'Frais', 'Oxydé', 'Transition', 'Frais']
    })
    
    st.dataframe(example_data)
    
    # Télécharger l'exemple
    csv_example = example_data.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger l'exemple",
        data=csv_example,
        file_name="exemple_donnees_or.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #1E2329; border-radius: 10px;'>
        <p style='color: #FFD700; font-size: 1.1rem; margin: 0;'>
            🏆 Gold Analysis Pro v2.1
        </p>
        <p style='color: #B8BCC0; margin: 5px 0;'>
            Développé par Didier Ouedraogo, P.Geo
        </p>
        <p style='color: #6C757D; font-size: 0.9rem;'>
            © 2025 - Analyse Avancée des Données Aurifères
        </p>
    </div>
""", unsafe_allow_html=True)