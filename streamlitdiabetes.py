import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import requests
from io import StringIO

# Configuración de la página
st.set_page_config(
    page_title="Análisis ML - Diabetes",
    page_icon="🏥",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #EFF6FF, #E0E7FF);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar datos
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv'
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

# Función para entrenar el modelo
@st.cache_resource
def train_model(data):
    # Separar características y target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo de Regresión Logística
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return model, scaler, metrics, X_train, X_test, y_train, y_test

# Cargar datos
data = load_data()
model, scaler, metrics, X_train, X_test, y_train, y_test = train_model(data)

# Título principal
st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;'>
        <h1 style='color: #1F2937; margin: 0;'>🏥 Análisis de Machine Learning - Diabetes</h1>
        <p style='color: #6B7280; margin: 10px 0 0 0;'>Dataset Pima Indians Diabetes - Modelo de Clasificación</p>
    </div>
""", unsafe_allow_html=True)

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🤖 Modelo y Métricas", "🔮 Predicción"])

# TAB 1: Análisis Exploratorio
with tab1:
    st.markdown("### Análisis Exploratorio de Datos")
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = len(data)
    diabetes_count = data[data['Outcome'] == 1].shape[0]
    no_diabetes_count = data[data['Outcome'] == 0].shape[0]
    avg_age = data['Age'].mean()
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color: #6B7280; font-size: 14px; margin: 0;'>Total de Casos</p>
                <p style='color: #1F2937; font-size: 28px; font-weight: bold; margin: 5px 0 0 0;'>{total_cases}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color: #6B7280; font-size: 14px; margin: 0;'>Con Diabetes</p>
                <p style='color: #EF4444; font-size: 28px; font-weight: bold; margin: 5px 0 0 0;'>{diabetes_count}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color: #6B7280; font-size: 14px; margin: 0;'>Sin Diabetes</p>
                <p style='color: #3B82F6; font-size: 28px; font-weight: bold; margin: 5px 0 0 0;'>{no_diabetes_count}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color: #6B7280; font-size: 14px; margin: 0;'>Edad Promedio</p>
                <p style='color: #8B5CF6; font-size: 28px; font-weight: bold; margin: 5px 0 0 0;'>{avg_age:.1f}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de casos
        outcome_dist = data['Outcome'].value_counts().reset_index()
        outcome_dist.columns = ['Diagnóstico', 'Cantidad']
        outcome_dist['Diagnóstico'] = outcome_dist['Diagnóstico'].map({0: 'Sin Diabetes', 1: 'Con Diabetes'})
        
        fig_outcome = px.bar(
            outcome_dist, 
            x='Diagnóstico', 
            y='Cantidad',
            title='Distribución de Casos',
            color='Diagnóstico',
            color_discrete_map={'Sin Diabetes': '#3B82F6', 'Con Diabetes': '#EF4444'},
            text='Cantidad'
        )
        fig_outcome.update_layout(showlegend=False, height=400)
        fig_outcome.update_traces(textposition='outside')
        st.plotly_chart(fig_outcome, use_container_width=True)
    
    with col2:
        # Distribución por edad
        age_bins = [20, 30, 40, 50, 60, 100]
        age_labels = ['20-30', '31-40', '41-50', '51-60', '61+']
        data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)
        age_dist = data['AgeGroup'].value_counts().sort_index().reset_index()
        age_dist.columns = ['Rango de Edad', 'Cantidad']
        
        fig_age = px.bar(
            age_dist, 
            x='Rango de Edad', 
            y='Cantidad',
            title='Distribución por Edad',
            color_discrete_sequence=['#8B5CF6'],
            text='Cantidad'
        )
        fig_age.update_layout(height=400)
        fig_age.update_traces(textposition='outside')
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Scatter plot
    st.markdown("### Relación Glucosa vs BMI")
    fig_scatter = px.scatter(
        data.sample(200), 
        x='Glucose', 
        y='BMI',
        color='Outcome',
        color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
        labels={'Outcome': 'Diagnóstico'},
        title='Glucosa vs BMI (Coloreado por Diagnóstico)',
        opacity=0.6
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Matriz de correlación
    st.markdown("### Matriz de Correlación")
    corr_matrix = data.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Correlación entre Variables'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB 2: Modelo y Métricas
with tab2:
    st.markdown("### Métricas del Modelo")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ('Accuracy', metrics['accuracy'], '🎯', '#3B82F6'),
        ('Precision', metrics['precision'], '🧠', '#10B981'),
        ('Recall', metrics['recall'], '📈', '#8B5CF6'),
        ('F1-Score', metrics['f1_score'], '⚡', '#F59E0B')
    ]
    
    for col, (name, value, icon, color) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            percentage = value * 100
            st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                        <p style='color: #6B7280; font-size: 14px; margin: 0;'>{name}</p>
                        <span style='font-size: 20px;'>{icon}</span>
                    </div>
                    <p style='color: {color}; font-size: 32px; font-weight: bold; margin: 0;'>{percentage:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Matriz de confusión
        st.markdown("### Matriz de Confusión")
        cm = metrics['confusion_matrix']
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicho: No', 'Predicho: Sí'],
            y=['Real: No', 'Real: Sí'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues',
            showscale=False
        ))
        fig_cm.update_layout(
            title='Matriz de Confusión',
            height=400,
            xaxis_title='Predicción',
            yaxis_title='Valor Real'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Importancia de características
        st.markdown("### Importancia de Características")
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(model.coef_[0])
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.barh(
            feature_importance,
            x='Importance',
            y='Feature',
            title='Importancia de Características',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Información del modelo
    st.markdown("### Información del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 4px solid #3B82F6;'>
                <p style='color: #6B7280; font-size: 14px; margin: 0 0 5px 0;'>Tipo de Modelo</p>
                <p style='color: #1F2937; font-size: 16px; font-weight: 600; margin: 0;'>Regresión Logística</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        num_features = len(X_train.columns)
        st.markdown(f"""
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 4px solid #8B5CF6;'>
                <p style='color: #6B7280; font-size: 14px; margin: 0 0 5px 0;'>Características</p>
                <p style='color: #1F2937; font-size: 16px; font-weight: 600; margin: 0;'>{num_features} variables predictoras</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        train_size = len(X_train)
        test_size = len(X_test)
        st.markdown(f"""
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 4px solid #10B981;'>
                <p style='color: #6B7280; font-size: 14px; margin: 0 0 5px 0;'>División del Dataset</p>
                <p style='color: #1F2937; font-size: 16px; font-weight: 600; margin: 0;'>80% Train ({train_size}), 20% Test ({test_size})</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 4px solid #F59E0B;'>
                <p style='color: #6B7280; font-size: 14px; margin: 0 0 5px 0;'>Normalización</p>
                <p style='color: #1F2937; font-size: 16px; font-weight: 600; margin: 0;'>Standard Scaler (Z-Score)</p>
            </div>
        """, unsafe_allow_html=True)

# TAB 3: Predicción
with tab3:
    st.markdown("### 🔮 Realizar Predicción")
    
    st.markdown("""
        <div style='background: #DBEAFE; padding: 15px; border-radius: 10px; border-left: 4px solid #3B82F6; margin-bottom: 20px;'>
            <p style='color: #1E40AF; margin: 0; font-size: 14px;'>
                <strong>Instrucciones:</strong> Ingrese los valores para cada característica y presione el botón para obtener una predicción.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Formulario de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Embarazos', min_value=0, max_value=20, value=3, step=1)
        glucose = st.number_input('Glucosa (mg/dL)', min_value=0, max_value=300, value=120, step=1)
        blood_pressure = st.number_input('Presión Arterial (mm Hg)', min_value=0, max_value=200, value=70, step=1)
        skin_thickness = st.number_input('Grosor de Piel (mm)', min_value=0, max_value=100, value=20, step=1)
    
    with col2:
        insulin = st.number_input('Insulina (μU/mL)', min_value=0, max_value=900, value=79, step=1)
        bmi = st.number_input('BMI (Índice de Masa Corporal)', min_value=0.0, max_value=70.0, value=32.0, step=0.1)
        dpf = st.number_input('Función de Pedigree de Diabetes', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input('Edad (años)', min_value=18, max_value=120, value=33, step=1)
    
    # Botón de predicción
    if st.button('🎯 Predecir Diagnóstico', use_container_width=True):
        # Preparar datos para predicción
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        # Escalar datos
        input_scaled = scaler.transform(input_data)
        
        # Realizar predicción
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        prob_diabetes = probability[1] * 100
        prob_no_diabetes = probability[0] * 100
        
        # Mostrar resultado
        if prediction == 1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                            padding: 30px; border-radius: 15px; color: white; margin-top: 20px;
                            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);'>
                    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 15px;'>
                        <span style='font-size: 48px;'>⚠️</span>
                        <div>
                            <h2 style='margin: 0; font-size: 28px;'>Alto Riesgo de Diabetes</h2>
                            <p style='margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;'>
                                Probabilidad: {prob_diabetes:.1f}%
                            </p>
                        </div>
                    </div>
                    <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;'>
                        <p style='margin: 0; font-size: 14px;'>
                            Se recomienda consultar con un profesional médico para evaluación y diagnóstico definitivo.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                            padding: 30px; border-radius: 15px; color: white; margin-top: 20px;
                            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);'>
                    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 15px;'>
                        <span style='font-size: 48px;'>✅</span>
                        <div>
                            <h2 style='margin: 0; font-size: 28px;'>Bajo Riesgo de Diabetes</h2>
                            <p style='margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;'>
                                Probabilidad de no diabetes: {prob_no_diabetes:.1f}%
                            </p>
                        </div>
                    </div>
                    <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;'>
                        <p style='margin: 0; font-size: 14px;'>
                            Continúe con hábitos saludables y chequeos médicos regulares.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Mostrar probabilidades
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                    <p style='color: #6B7280; font-size: 14px; margin: 0 0 10px 0;'>Probabilidad Sin Diabetes</p>
                    <div style='background: #E0E7FF; height: 30px; border-radius: 15px; overflow: hidden;'>
                        <div style='background: #3B82F6; height: 100%; width: {prob_no_diabetes}%; 
                                    display: flex; align-items: center; justify-content: center;'>
                            <span style='color: white; font-weight: bold; font-size: 12px;'>{prob_no_diabetes:.1f}%</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                    <p style='color: #6B7280; font-size: 14px; margin: 0 0 10px 0;'>Probabilidad Con Diabetes</p>
                    <div style='background: #FEE2E2; height: 30px; border-radius: 15px; overflow: hidden;'>
                        <div style='background: #EF4444; height: 100%; width: {prob_diabetes}%; 
                                    display: flex; align-items: center; justify-content: center;'>
                            <span style='color: white; font-weight: bold; font-size: 12px;'>{prob_diabetes:.1f}%</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Nota importante
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: #FEF3C7; padding: 20px; border-radius: 10px; border-left: 4px solid #F59E0B;'>
            <h4 style='color: #92400E; margin: 0 0 10px 0;'>⚠️ Nota Importante</h4>
            <p style='color: #78350F; margin: 0; font-size: 14px;'>
                Este modelo es con fines <strong>educativos y demostrativos</strong>. Las predicciones no deben ser 
                utilizadas como diagnóstico médico real. Siempre consulte con profesionales de la salud calificados 
                para obtener un diagnóstico y tratamiento adecuado.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
total_data = len(data)
accuracy_pct = metrics['accuracy'] * 100
st.markdown(f"""
    <div style='text-align: center; color: #6B7280; padding: 20px;'>
        <p style='margin: 0;'>🏥 Análisis de Machine Learning - Dataset Pima Indians Diabetes</p>
        <p style='margin: 5px 0 0 0; font-size: 12px;'>Modelo entrenado con {total_data} registros | Accuracy: {accuracy_pct:.2f}%</p>
    </div>
""", unsafe_allow_html=True)
