import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def predict_sentiment(text):
    """
    Analiza el sentimiento de un texto
    """
    text = text.lower()
    
    # Diccionario de expresiones
    expressions = {
        'muy_negativo': [
            'pésimo', 'horrible', 'terrible', 'malísimo',
            'nunca más', 'no recomiendo', 'muy malo',
            'pésima atención', 'mala experiencia'
        ],
        'muy_positivo': [
            'excelente', 'increíble', 'perfecto', 'maravilloso',
            'el mejor', 'muy recomendado', 'espectacular',
            'excelente servicio', 'muy buena experiencia'
        ],
        'negativo': [
            'malo', 'regular', 'deficiente', 'no vale',
            'caro', 'lento', 'mala', 'no me gustó'
        ],
        'positivo': [
            'bueno', 'recomendable', 'agradable', 'buen servicio',
            'buena atención', 'me gustó', 'volvería'
        ]
    }
    
    # Detectar negaciones
    negation_words = ['no', 'nunca', 'ni', 'tampoco', 'nada']
    has_negation = any(word in text.split() for word in negation_words)
    
    # Análisis
    if any(expr in text for expr in expressions['muy_negativo']):
        sentiment = 1
    elif any(expr in text for expr in expressions['muy_positivo']):
        sentiment = 5 if not has_negation else 2
    elif has_negation:
        sentiment = 2
    elif any(expr in text for expr in expressions['positivo']):
        sentiment = 4
    elif any(expr in text for expr in expressions['negativo']):
        sentiment = 2
    else:
        sentiment = 3
    
    return {
        'texto': text,
        'sentimiento': sentiment,
        'interpretacion': {
            1: "Muy negativo",
            2: "Negativo",
            3: "Neutral",
            4: "Positivo",
            5: "Muy positivo"
        }[sentiment]
    }

def load_sentiment_data():
    """
    Carga los datos de análisis de sentimientos
    """
    try:
        # Cargar datos de sentimientos
        df_sentiment = pd.read_csv('sentiment_results.csv')
        
        # Verificar que tenemos todas las columnas necesarias
        required_columns = ['texto', 'sentiment', 'sentiment_label', 'label', 'accuracy']
        if not all(col in df_sentiment.columns for col in required_columns):
            st.warning("Algunas columnas están faltando en el archivo de datos")
        
        return df_sentiment
    except Exception as e:
        st.error(f"Error cargando datos de sentimientos: {str(e)}")
        return None

def load_forecast_data():
    """
    Carga los datos de pronóstico del notebook 2_forecasting
    """
    try:
        # Cargar datos de pronóstico
        df_forecast = pd.read_csv('forecast_results.csv')
        return df_forecast
    except Exception as e:
        st.error(f"Error cargando datos de pronóstico: {str(e)}")
        return None

def show_general_summary():
    st.header("Resumen General")
    
    # Cargar datos
    df_sentiment = load_sentiment_data()
    df_forecast = load_forecast_data()
    
    st.markdown("""
    ### 📌 Hallazgos Principales
    Análisis integrado de sentimientos y pronósticos de demanda energética.
    """)

    # Métricas de sentimientos
    if df_sentiment is not None:
        st.subheader("Análisis de Sentimientos")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precisión del Modelo", 
                     f"{df_sentiment['accuracy'].mean():.2%}")
        with col2:
            st.metric("Total Reseñas", 
                     len(df_sentiment))
        with col3:
            st.metric("Sentimiento Promedio", 
                     f"{df_sentiment['sentiment'].mean():.1f}/5")
    
    # Métricas de pronóstico
    if df_forecast is not None:
        st.subheader("Pronóstico de Demanda")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", 
                     f"{df_forecast['mae'].mean():.4f} GW")
        with col2:
            st.metric("RMSE", 
                     f"{df_forecast['rmse'].mean():.4f} GW")
        with col3:
            st.metric("MAPE", 
                     f"{df_forecast['mape'].mean():.2f}%")

def show_time_series():
    st.header("Análisis de Serie Temporal de Demanda Energética")
    
    # Cargar datos de pronóstico
    df_forecast = load_forecast_data()
    
    if df_forecast is not None:
        try:
            # Asegurar que la columna fecha sea datetime
            df_forecast['fecha'] = pd.to_datetime(df_forecast['fecha'])
            
            # Filtros de fecha
            col1, col2 = st.columns(2)
            with col1:
                start_date = pd.to_datetime(st.date_input("Fecha inicial", 
                                          df_forecast['fecha'].min().date()))
            with col2:
                end_date = pd.to_datetime(st.date_input("Fecha final", 
                                        df_forecast['fecha'].max().date()))
            
            # Filtrar datos
            mask = (df_forecast['fecha'] >= start_date) & \
                   (df_forecast['fecha'] <= end_date)
            df_filtered = df_forecast.loc[mask]
            
            # Visualizaciones
            st.subheader("Predicciones vs Valores Reales")
            fig1 = px.line(df_filtered, 
                          x='fecha', 
                          y=['valor_real', 'prediccion'],
                          title='Demanda Energética: Valores Reales vs Predicciones',
                          labels={
                              'valor_real': 'Demanda Real (GW)',
                              'prediccion': 'Predicción (GW)',
                              'fecha': 'Fecha'
                          })
            st.plotly_chart(fig1)
            
            # Métricas de error
            st.subheader("Métricas de Error")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{df_filtered['mae'].mean():.4f} GW")
            with col2:
                st.metric("RMSE", f"{df_filtered['rmse'].mean():.4f} GW")
            with col3:
                st.metric("MAPE", f"{df_filtered['mape'].mean():.2f}%")
            
            # Análisis de tendencia
            st.subheader("Análisis de Tendencia")
            df_filtered['error_absoluto'] = abs(df_filtered['valor_real'] - df_filtered['prediccion'])
            df_filtered['error_porcentual'] = (df_filtered['error_absoluto'] / df_filtered['valor_real']) * 100
            
            # Resumen estadístico
            st.write("Resumen Estadístico de la Demanda (GW)")
            stats_df = pd.DataFrame({
                'Métrica': ['Media', 'Mediana', 'Máximo', 'Mínimo', 'Desv. Estándar'],
                'Valor Real': [
                    f"{df_filtered['valor_real'].mean():.2f}",
                    f"{df_filtered['valor_real'].median():.2f}",
                    f"{df_filtered['valor_real'].max():.2f}",
                    f"{df_filtered['valor_real'].min():.2f}",
                    f"{df_filtered['valor_real'].std():.2f}"
                ],
                'Predicción': [
                    f"{df_filtered['prediccion'].mean():.2f}",
                    f"{df_filtered['prediccion'].median():.2f}",
                    f"{df_filtered['prediccion'].max():.2f}",
                    f"{df_filtered['prediccion'].min():.2f}",
                    f"{df_filtered['prediccion'].std():.2f}"
                ]
            })
            st.table(stats_df)
            
            # Distribución del error
            st.subheader("Distribución del Error de Predicción")
            fig2 = px.histogram(df_filtered, 
                              x='error_porcentual',
                              nbins=30,
                              title='Distribución del Error Porcentual',
                              labels={'error_porcentual': 'Error (%)'})
            st.plotly_chart(fig2)
            
        except Exception as e:
            st.error(f"Error procesando datos: {str(e)}")
            st.write("Detalles del error:", e)

def show_sentiment_analysis():
    st.header("Análisis de Sentimientos")
    
    # Cargar datos de sentimientos
    df_sentiment = load_sentiment_data()
    
    if df_sentiment is not None:
        try:
            # Convertir label a numérico si es necesario
            if df_sentiment['label'].dtype == 'object':
                df_sentiment['label'] = pd.to_numeric(df_sentiment['label'])
            
            # Mostrar métricas generales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Reseñas", len(df_sentiment))
            with col2:
                accuracy = df_sentiment['accuracy'].mean()
                st.metric("Precisión del Modelo", f"{accuracy:.2%}")
            with col3:
                sentiment_avg = df_sentiment['sentiment'].mean()
                st.metric("Sentimiento Promedio", f"{sentiment_avg:.2f}/5")
            
            # Distribución de sentimientos
            st.subheader("Distribución de Sentimientos")
            fig = px.pie(df_sentiment, 
                        names='sentiment_label',
                        title='Distribución de Sentimientos en Reseñas')
            st.plotly_chart(fig)
            
            # Comparación predicción vs realidad
            st.subheader("Predicciones vs Etiquetas Reales")
            
            # Asegurar que las etiquetas estén en el mismo formato
            label_mapping = {
                1: "Muy negativo",
                2: "Negativo",
                3: "Neutral",
                4: "Positivo",
                5: "Muy positivo"
            }
            
            # Convertir las etiquetas numéricas a texto
            df_sentiment['label_text'] = df_sentiment['label'].astype(int).map(label_mapping)
            
            comparison_df = pd.DataFrame({
                'Predicción': df_sentiment['sentiment_label'].value_counts(),
                'Real': df_sentiment['label_text'].value_counts()
            }).fillna(0)
            
            # Asegurar que tenemos todas las categorías
            for categoria in label_mapping.values():
                if categoria not in comparison_df.index:
                    comparison_df.loc[categoria] = [0, 0]
            
            # Ordenar las categorías
            comparison_df = comparison_df.reindex(label_mapping.values())
            
            fig2 = px.bar(comparison_df, 
                         barmode='group',
                         title='Comparación de Predicciones vs Valores Reales')
            st.plotly_chart(fig2)
            
            # Agregar sección de prueba del modelo
            st.subheader("🔍 Probar el Modelo")
            
            # Input de texto
            user_input = st.text_area(
                "Ingrese un texto para analizar:",
                placeholder="Ejemplo: El servicio fue excelente, muy recomendado"
            )
            
            # Botón de análisis
            if st.button("Analizar Sentimiento"):
                if user_input:
                    resultado = predict_sentiment(user_input)
                    
                    # Mostrar resultado
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Sentimiento detectado:** {resultado['interpretacion']}")
                        st.metric("Nivel", f"{resultado['sentimiento']}/5")
                    
                    with col2:
                        # Gauge chart para visualización
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = resultado['sentimiento'],
                            title = {'text': "Nivel de Sentimiento"},
                            gauge = {
                                'axis': {'range': [1, 5]},
                                'steps': [
                                    {'range': [1, 2], 'color': "lightcoral"},
                                    {'range': [2, 3], 'color': "lightyellow"},
                                    {'range': [3, 4], 'color': "lightgreen"},
                                    {'range': [4, 5], 'color': "darkgreen"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)
            
            # Ejemplos predefinidos
            st.subheader("📝 Ejemplos")
            ejemplos = [
                "Excelente servicio, muy recomendado",
                "La atención fue terrible, no volvería",
                "Normal, el precio es razonable",
                "Buena atención pero muy caro",
                "No me gustó la experiencia"
            ]
            
            for ejemplo in ejemplos:
                if st.button(f"Probar: {ejemplo}"):
                    resultado = predict_sentiment(ejemplo)
                    st.info(f"Sentimiento: {resultado['interpretacion']} ({resultado['sentimiento']}/5)")
            
        except Exception as e:
            st.error(f"Error en el análisis de sentimientos: {str(e)}")
            st.write("Detalles del error:", e)

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Análisis de Sentimientos y Pronósticos",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("📊 Dashboard: Análisis de Sentimientos y Pronósticos")
    st.markdown("""
    Este dashboard integra el análisis de sentimientos de reseñas con 
    pronósticos de series temporales, mostrando métricas y tendencias clave.
    """)

    # Sidebar
    st.sidebar.title("Navegación")
    st.sidebar.markdown("""
    ### Sobre este Dashboard
    Integración de análisis de sentimientos y pronósticos temporales.
    """)
    
    page = st.sidebar.radio(
        "Seleccione una sección:",
        ["Resumen General", "Serie Temporal", "Análisis de Sentimientos"]
    )

    if page == "Resumen General":
        show_general_summary()
    elif page == "Serie Temporal":
        show_time_series()
    else:
        show_sentiment_analysis()

if __name__ == "__main__":
    main()