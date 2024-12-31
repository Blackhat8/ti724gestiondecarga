import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time
import io
import random

# Configuración de la página
st.set_page_config(
    page_title="TI724 Gestión de Cargas de Trabajo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Datos de empleados y proyectos
def generate_base_data():
    empleados = {
        'Empleado': [
            'Carlos Martínez', 'Ana García', 'Miguel Rodríguez', 'Laura Sánchez',
            'David López', 'Isabel Torres', 'Roberto Díaz', 'Patricia Ruiz',
            'Fernando Morales', 'Elena Castro', 'Juan Vargas', 'María González'
        ],
        'Skills': [
            'Python,AWS,React', 'Java,Docker,SQL', 'Python,ML,Data Science',
            'JavaScript,React,Node', 'Python,DevOps,AWS', 'Java,Spring,SQL',
            'React,Angular,Vue', 'Python,Django,PostgreSQL', 'AWS,Terraform,Docker',
            'ML,TensorFlow,Python', 'Node,Express,MongoDB', 'React,TypeScript,Redux'
        ],
        'FTE': [
            0.8, 0.7, 0.9, 0.75, 0.6, 0.85, 0.5, 0.95, 0.8, 0.7, 0.9, 0.85
        ],
        'Riesgo_Burnout': [
            0.3, 0.6, 0.4, 0.5, 0.7, 0.3, 0.2, 0.6, 0.5, 0.4, 0.5, 0.4
        ],
        'Proyecto_Actual': [
            'Portal Bancario', 'Sistema ERP', 'Análisis Predictivo',
            'App Móvil', 'Infraestructura Cloud', 'Sistema ERP',
            'Portal Bancario', 'App Móvil', 'Infraestructura Cloud',
            'Análisis Predictivo', 'Portal Bancario', 'App Móvil'
        ],
        'Productividad': [
            85, 78, 92, 88, 76, 82, 79, 91, 87, 83, 86, 90
        ]
    }
    
    proyectos = {
        'Proyecto': [
            'Portal Bancario', 'Sistema ERP', 'Análisis Predictivo',
            'App Móvil', 'Infraestructura Cloud'
        ],
        'Progreso_Real': [0.75, 0.60, 0.85, 0.45, 0.90],
        'Progreso_Planificado': [0.80, 0.65, 0.80, 0.50, 0.85],
        'Fecha_Inicio': [
            '2023-09-01', '2023-10-15', '2023-11-01',
            '2023-12-01', '2023-08-15'
        ],
        'Fecha_Fin': [
            '2024-03-01', '2024-05-15', '2024-04-01',
            '2024-06-01', '2024-02-15'
        ],
        'Complejidad': [0.8, 0.7, 0.9, 0.6, 0.85]
    }
    
    df_empleados = pd.DataFrame(empleados)
    df_proyectos = pd.DataFrame(proyectos)
    
    return df_empleados, df_proyectos

# Función para sugerir empleados para nuevos proyectos usando ML
def sugerir_empleados_ml(skills_requeridos, horas_requeridas, df_empleados):
    # Crear matriz de similitud de skills
    all_skills = set(','.join(df_empleados['Skills']).split(','))
    skill_matrix = []
    
    for emp_skills in df_empleados['Skills']:
        emp_skill_vector = [1 if skill in emp_skills.split(',') else 0 for skill in all_skills]
        skill_matrix.append(emp_skill_vector)
    
    required_skill_vector = [1 if skill in skills_requeridos.split(',') else 0 for skill in all_skills]
    
    # Calcular similitud
    similarity_scores = cosine_similarity([required_skill_vector], skill_matrix)[0]
    
    # Combinar con disponibilidad, riesgo de burnout y productividad
    availability_scores = 1 - df_empleados['FTE']
    burnout_safety = 1 - df_empleados['Riesgo_Burnout']
    productivity_scores = df_empleados['Productividad'] / 100  # Normalizar productividad
    
    # Usar KMeans para agrupar empleados
    features = np.column_stack((similarity_scores, availability_scores, burnout_safety, productivity_scores))
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Calcular puntuación final
    final_scores = similarity_scores * availability_scores * burnout_safety * productivity_scores
    
    # Obtener los mejores candidatos de cada cluster
    best_candidates = []
    for cluster in range(3):
        cluster_candidates = df_empleados[clusters == cluster]
        cluster_scores = final_scores[clusters == cluster]
        if len(cluster_candidates) > 0:
            best_in_cluster = cluster_candidates.iloc[cluster_scores.argmax()]
            best_candidates.append(best_in_cluster)
    
    return pd.DataFrame(best_candidates)

# Función para predecir el riesgo de burnout
def predecir_riesgo_burnout(df_empleados):
    if df_empleados.empty or len(df_empleados) < 2:
        return df_empleados
    
    # Preparar los datos
    X = df_empleados[['FTE', 'Productividad']]
    y = df_empleados['Riesgo_Burnout']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predecir el riesgo de burnout para todos los empleados
    df_empleados['Riesgo_Burnout_Predicho'] = model.predict(X)
    
    return df_empleados

# Función para generar alertas
def generar_alertas(df_empleados, df_proyectos):
    alertas = []
    
    if not df_empleados.empty:
        # Alerta de sobrecarga
        sobrecargados = df_empleados[df_empleados['FTE'] > 0.8]
        for _, emp in sobrecargados.iterrows():
            alertas.append(f"⚠️ {emp['Empleado']} está sobrecargado (FTE: {emp['FTE']:.2f})")
    
        # Alerta de riesgo de burnout
        alto_riesgo = df_empleados[df_empleados['Riesgo_Burnout_Predicho'] > 0.6]
        for _, emp in alto_riesgo.iterrows():
            alertas.append(f"🔥 {emp['Empleado']} tiene alto riesgo de burnout (Riesgo predicho: {emp['Riesgo_Burnout_Predicho']:.2f})")
    
    if not df_proyectos.empty:
        # Alerta de proyectos atrasados
        atrasados = df_proyectos[df_proyectos['Progreso_Real'] < df_proyectos['Progreso_Planificado'] - 0.1]
        for _, proj in atrasados.iterrows():
            alertas.append(f"📉 El proyecto {proj['Proyecto']} está atrasado (Real: {proj['Progreso_Real']:.2f}, Planificado: {proj['Progreso_Planificado']:.2f})")
    
    return alertas

# Estilos CSS personalizados
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    .alert-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .alert-info {
        background-color: #cce5ff;
        border: 1px solid #b8daff;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicialización del estado de la sesión
if 'df_empleados' not in st.session_state or 'df_proyectos' not in st.session_state:
    st.session_state.df_empleados, st.session_state.df_proyectos = generate_base_data()
    st.session_state.df_empleados = predecir_riesgo_burnout(st.session_state.df_empleados)

# Función para actualizar datos y mostrar notificaciones
def actualizar_datos(mensaje):
    st.session_state.df_empleados = predecir_riesgo_burnout(st.session_state.df_empleados)
    st.toast(mensaje, icon="🔔")
    st.rerun()

# Título principal con animación
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5; animation: fadeIn 1.5s;'>
        🚀 TI724 Gestión de Cargas de Trabajo
    </h1>
    """, unsafe_allow_html=True)

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Principal",
    "👥 Gestión de Recursos",
    "📈 Seguimiento de Proyectos",
    "🎯 Planificación",
    "🔔 Alertas y Notificaciones"
])

with tab1:
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Empleados Activos", len(st.session_state.df_empleados))
    with col2:
        st.metric("Proyectos en Curso", len(st.session_state.df_proyectos))
    with col3:
        st.metric("FTE Promedio", f"{st.session_state.df_empleados['FTE'].mean():.2f}" if not st.session_state.df_empleados.empty else "N/A")
    with col4:
        st.metric("Riesgo Promedio", f"{st.session_state.df_empleados['Riesgo_Burnout_Predicho'].mean():.2f}" if not st.session_state.df_empleados.empty else "N/A")
    
    # Gráfico de distribución FTE
    st.subheader("📊 Distribución de Cargas de Trabajo")
    if not st.session_state.df_empleados.empty:
        fig_fte = px.bar(
            st.session_state.df_empleados,
            x="Empleado",
            y="FTE",
            color="Riesgo_Burnout_Predicho",
            color_continuous_scale="RdYlBu_r",
            title="FTE por Empleado"
        )
        st.plotly_chart(fig_fte, use_container_width=True)
    else:
        st.info("No hay datos de empleados disponibles.")
    
    # Tabla de alertas
    st.subheader("⚠️ Alertas Activas")
    alertas = generar_alertas(st.session_state.df_empleados, st.session_state.df_proyectos)
    if alertas:
        for alerta in alertas:
            if "sobrecargado" in alerta:
                st.markdown(f'<div class="alert-box alert-warning">{alerta}</div>', unsafe_allow_html=True)
            elif "burnout" in alerta:
                st.markdown(f'<div class="alert-box alert-danger">{alerta}</div>', unsafe_allow_html=True)
            elif "atrasado" in alerta:
                st.markdown(f'<div class="alert-box alert-info">{alerta}</div>', unsafe_allow_html=True)
    else:
        st.success("No hay alertas activas en este momento.")

with tab2:
    st.subheader("👥 Gestión de Recursos Humanos")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        filtro_proyecto = st.selectbox(
            "Filtrar por Proyecto",
            ["Todos"] + list(st.session_state.df_empleados['Proyecto_Actual'].unique()) if not st.session_state.df_empleados.empty else ["Todos"]
        )
    with col2:
        filtro_skill = st.multiselect(
            "Filtrar por Skills",
            list(set(','.join(st.session_state.df_empleados['Skills']).split(','))) if not st.session_state.df_empleados.empty else []
        )
    
    # Aplicar filtros
    df_filtrado = st.session_state.df_empleados.copy()
    if not st.session_state.df_empleados.empty:
        if filtro_proyecto != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Proyecto_Actual'] == filtro_proyecto]
        if filtro_skill:
            df_filtrado = df_filtrado[df_filtrado['Skills'].apply(
                lambda x: any(skill in x for skill in filtro_skill)
            )]
        
        # Mostrar tabla interactiva
        st.dataframe(
            df_filtrado,
            hide_index=True,
            column_config={
                "FTE": st.column_config.ProgressColumn(
                    "Carga de Trabajo",
                    help="Full Time Equivalent",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Riesgo_Burnout_Predicho": st.column_config.ProgressColumn(
                    "Riesgo Predicho",
                    help="Nivel de riesgo de burnout predicho",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                )
            }
        )
    else:
        st.info("No hay datos de empleados disponibles.")

with tab3:
    st.subheader("📈 Estado de Proyectos")
    
    if not st.session_state.df_proyectos.empty:
        # Gráfico de progreso
        fig_proyectos = go.Figure()
        
        fig_proyectos.add_trace(go.Bar(
            name='Progreso Real',
            x=st.session_state.df_proyectos['Proyecto'],
            y=st.session_state.df_proyectos['Progreso_Real'],
            marker_color='rgb(26, 118, 255)'
        ))
        
        fig_proyectos.add_trace(go.Bar(
            name='Progreso Planificado',
            x=st.session_state.df_proyectos['Proyecto'],
            y=st.session_state.df_proyectos['Progreso_Planificado'],
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig_proyectos.update_layout(
            barmode='group',
            title='Progreso de Proyectos: Real vs Planificado'
        )
        
        st.plotly_chart(fig_proyectos, use_container_width=True)
        
        # Tabla de proyectos
        st.session_state.df_proyectos['Desviación'] = (
            st.session_state.df_proyectos['Progreso_Real'] - st.session_state.df_proyectos['Progreso_Planificado']
        ).round(2)
        
        st.dataframe(
            st.session_state.df_proyectos,
            hide_index=True,
            column_config={
                "Progreso_Real": st.column_config.ProgressColumn(
                    "Progreso Real",
                    help="Progreso actual del proyecto",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Progreso_Planificado": st.column_config.ProgressColumn(
                    "Progreso Planificado",
                    help="Progreso esperado del proyecto",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                )
            }
        )
    else:
        st.info("No hay datos de proyectos disponibles.")

with tab4:
    st.subheader("🎯 Planificación de Nuevos Proyectos")
    
    # Inicializar sugerencias como None
    if 'sugerencias' not in st.session_state:
        st.session_state.sugerencias = None
    
    # Formulario para nuevo proyecto
    with st.form("nuevo_proyecto"):
        col1, col2 = st.columns(2)
        with col1:
            nombre_proyecto = st.text_input("Nombre del Proyecto")
            skills_requeridos = st.multiselect(
                "Skills Requeridos",
                list(set(','.join(st.session_state.df_empleados['Skills']).split(','))) if not st.session_state.df_empleados.empty else []
            )
        with col2:
            horas_requeridas = st.number_input("Horas Requeridas", min_value=1, value=40)
            fecha_inicio = st.date_input("Fecha de Inicio")
        
        submitted = st.form_submit_button("Buscar Recursos Disponibles")
        
        if submitted and skills_requeridos:
            with st.spinner("Analizando recursos disponibles..."):
                time.sleep(1)  # Simular procesamiento
                st.session_state.sugerencias = sugerir_empleados_ml(
                    ','.join(skills_requeridos),
                    horas_requeridas,
                    st.session_state.df_empleados
                )
                
                st.success("¡Análisis completado!")
                st.subheader("Recursos Sugeridos")
                
                st.dataframe(
                    st.session_state.sugerencias[['Empleado', 'Skills', 'FTE', 'Proyecto_Actual', 'Riesgo_Burnout_Predicho']],
                    hide_index=True,
                    column_config={
                        "FTE": st.column_config.ProgressColumn(
                            "Disponibilidad",
                            help="Capacidad disponible",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                        "Riesgo_Burnout_Predicho": st.column_config.ProgressColumn(
                            "Riesgo de Burnout",
                            help="Riesgo de burnout predicho",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        )
                    }
                )

    # Función para asignar empleado y actualizar datos
    def asignar_empleado(empleado, proyecto):
        if not st.session_state.df_empleados.empty:
            idx = st.session_state.df_empleados[st.session_state.df_empleados['Empleado'] == empleado].index[0]
            st.session_state.df_empleados.at[idx, 'FTE'] = min(st.session_state.df_empleados.at[idx, 'FTE'] + 0.2, 1.0)
            st.session_state.df_empleados.at[idx, 'Proyecto_Actual'] = proyecto
        
            nuevo_proyecto = {
                'Proyecto': proyecto,
                'Progreso_Real': 0.0,
                'Progreso_Planificado': 0.1,
                'Fecha_Inicio': fecha_inicio,
                'Fecha_Fin': fecha_inicio + timedelta(days=90),
                'Complejidad': random.uniform(0.5, 1.0)
            }
            st.session_state.df_proyectos = pd.concat([st.session_state.df_proyectos, pd.DataFrame([nuevo_proyecto])], ignore_index=True)
        
            actualizar_datos(f"Se ha asignado a {empleado} al proyecto {proyecto}")


    if st.session_state.sugerencias is not None and not st.session_state.sugerencias.empty:
        st.subheader("Asignación de Recursos")
        col1, col2 = st.columns(2)
        
        with col1:
            empleado_seleccionado = st.selectbox(
                "Seleccionar Empleado",
                options=st.session_state.sugerencias['Empleado'].tolist(),
                key="empleado_seleccionado"
            )
            if st.button("Asignar Manualmente"):
                asignar_empleado(empleado_seleccionado, nombre_proyecto)

        with col2:
            if st.button("Asignar Automáticamente"):
                empleado_asignado = st.session_state.sugerencias.iloc[0]['Empleado']
                asignar_empleado(empleado_asignado, nombre_proyecto)


with tab5:
    st.subheader("🔔 Alertas y Notificaciones")
    
    alertas = generar_alertas(st.session_state.df_empleados, st.session_state.df_proyectos)
    
    if alertas:
        for alerta in alertas:
            if "sobrecargado" in alerta:
                st.markdown(f'<div class="alert-box alert-warning">{alerta}</div>', unsafe_allow_html=True)
            elif "burnout" in alerta:
                st.markdown(f'<div class="alert-box alert-danger">{alerta}</div>', unsafe_allow_html=True)
            elif "atrasado" in alerta:
                st.markdown(f'<div class="alert-box alert-info">{alerta}</div>', unsafe_allow_html=True)
    else:
        st.success("No hay alertas activas en este momento.")

# Barra lateral con métricas de rendimiento
with st.sidebar:
    st.header("📊 Métricas del Sistema")
    st.metric("Tiempo de Respuesta", f"{random.uniform(0.5, 2.0):.2f}s")
    st.metric("Uso de CPU", f"{random.randint(30, 70)}%")
    st.metric("Memoria", f"{random.uniform(1.5, 3.0):.1f} GB")
    
    # Exportar datos
    st.subheader("📤 Exportar Datos")
    if st.button("Generar Reporte"):
        with st.spinner("Generando reporte..."):
            time.sleep(1)
            
            # Crear reporte
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                if not st.session_state.df_empleados.empty:
                    st.session_state.df_empleados.to_excel(writer, sheet_name='Empleados', index=False)
                if not st.session_state.df_proyectos.empty:
                    st.session_state.df_proyectos.to_excel(writer, sheet_name='Proyectos', index=False)
            
            st.download_button(
                label="📥 Descargar Reporte",
                data=buffer.getvalue(),
                file_name=f"reporte_ti724_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.ms-excel"
            )

# Pie de página
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2024 TI724. Todos los derechos reservados.</p>",
    unsafe_allow_html=True
)

