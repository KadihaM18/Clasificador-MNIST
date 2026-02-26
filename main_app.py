import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Configuración inicial
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

@st.cache_data
def load_mnist():
    # Cargamos una versión reducida para velocidad (10k muestras)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data[:10000], mnist.target[:10000]
    return X / 255.0, y  # Normalización

st.title("🔢 MNIST Hand-Drawn Digit Classifier")
st.markdown("Clasificador pedagógico de dígitos del 0 al 9 usando Machine Learning.")

# --- SIDEBAR: Configuración ---
st.sidebar.header("🛠️ Configuración")
model_choice = st.sidebar.selectbox(
    "Selecciona el Modelo:",
    ("Random Forest", "Logistic Regression", "SVM (RBF Kernel)")
)

test_size = st.sidebar.slider("Porcentaje de Test (%)", 10, 40, 20) / 100

# Botón para disparar entrenamiento
train_button = st.sidebar.button("🚀 Entrenar Modelo")

# --- CARGA Y PROCESAMIENTO ---
X, y = load_mnist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# --- ENTRENAMIENTO (Estado de Sesión) ---
if "model" not in st.session_state:
    st.session_state.model = None

if train_button:
    with st.spinner(f"Entrenando {model_choice}..."):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        else:
            model = SVC(probability=True)
            
        model.fit(X_train, y_train)
        st.session_state.model = model
        st.session_state.model_name = model_choice
        st.success("¡Modelo entrenado con éxito!")

# --- INTERFAZ PRINCIPAL ---
if st.session_state.model is not None:
    tab1, tab2 = st.tabs(["📊 Desempeño", "🎯 Predicción y Validación"])

    with tab1:
        st.header(f"Métricas: {st.session_state.model_name}")
        y_pred = st.session_state.model.predict(X_test)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy Final", f"{acc:.2%}")
            st.text("Reporte de Clasificación:")
            st.code(classification_report(y_test, y_pred))
        
        with col2:
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            st.pyplot(fig_cm)

    with tab2:
        st.header("Pruebas Individuales")
        st.write("Selecciona un índice del set de pruebas para ver el dígito y la predicción.")
        
        index = st.number_input("Índice de imagen (0 a 1000)", min_value=0, max_value=1000, value=0)
        
        test_img = X_test[index].reshape(28, 28)
        true_label = y_test[index]
        
        col_img, col_pred = st.columns(2)
        
        with col_img:
            st.subheader("Imagen del Dígito")
            fig, ax = plt.subplots()
            ax.imshow(test_img, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            st.caption(f"Etiqueta Real: {true_label}")

        with col_pred:
            st.subheader("Resultado del Modelo")
            prediction = st.session_state.model.predict(X_test[index].reshape(1, -1))[0]
            
            if prediction == true_label:
                st.balloons()
                st.success(f"### Predicción: {prediction}")
            else:
                st.error(f"### Predicción: {prediction} (Incorrecto)")
            
            # Gráfica de probabilidades
            probs = st.session_state.model.predict_proba(X_test[index].reshape(1, -1))[0]
            st.bar_chart(pd.DataFrame(probs, index=range(10), columns=["Probabilidad"]))

else:
    st.info("Presiona el botón en la barra lateral para entrenar el modelo y comenzar.")
