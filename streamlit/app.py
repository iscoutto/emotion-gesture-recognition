# O código a seguir cria uma aplicação Streamlit para detecção de emoções faciais.
# O aplicativo irá carregar o modelo Keras fornecido, usar a webcam para capturar
# o vídeo e exibir as emoções detectadas em tempo real.

# Importa as bibliotecas necessárias
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Título da aplicação Streamlit
st.title("Detecção de Emoções Faciais em Tempo Real")

# Define os rótulos de emoção, correspondendo exatamente ao seu notebook de teste
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# --- SEÇÃO DE CARREGAMENTO DO MODELO COM CACHING ---
@st.cache_resource
def load_keras_model():
    """Carrega o modelo Keras apenas uma vez."""
    try:
        model = load_model('mobilenetv2_finetuned_fer2013_v2.keras')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.info("Certifique-se de que o arquivo 'mobilenetv2_finetuned_fer2013_v2.keras' está na mesma pasta que este script.")
        st.stop()

# Carrega o modelo
model = load_keras_model()

# --- SEÇÃO DE CARREGAMENTO DO CLASSIFICADOR COM CACHING ---
@st.cache_resource
def load_haar_cascade():
    """Carrega o classificador Haar Cascade apenas uma vez."""
    try:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_classifier
    except Exception as e:
        st.error(f"Erro ao carregar o classificador Haar Cascade: {e}")
        st.stop()

# Carrega o classificador
face_classifier = load_haar_cascade()

# Variável de estado para controlar a captura de vídeo
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Área para exibir o feed de vídeo
frame_placeholder = st.empty()

# --- LÓGICA DO BOTÃO ÚNICO ---
if st.session_state.is_running:
    button_label = "Parar Câmera"
else:
    button_label = "Iniciar Câmera"

if st.button(button_label):
    st.session_state.is_running = not st.session_state.is_running

# --- LÓGICA DE CAPTURA DE VÍDEO ---
if st.session_state.is_running:
    # Acessa a câmera (0 é a webcam padrão)
    cap = cv2.VideoCapture(0)

    # Verifica se a câmera foi aberta com sucesso
    if not cap.isOpened():
        st.error("Erro: Não foi possível acessar a câmera.")
        st.session_state.is_running = False
    else:
        # Loop principal para processar o vídeo
        while st.session_state.is_running:
            # Captura um frame
            ret, frame = cap.read()

            if not ret:
                st.warning("Não foi possível ler o frame da câmera. Tentando novamente...")
                continue

            # Converte o frame para escala de cinza para detecção de rosto
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecta as faces no frame
            faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Para cada face detectada
            for (x, y, w, h) in faces:
                # Desenha um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extrai a região da face colorida
                roi_color = frame[y:y+h, x:x+w]
                
                # Redimensiona a face para 96x96, o tamanho esperado pelo seu modelo
                cropped_img = cv2.resize(roi_color, (96, 96), interpolation=cv2.INTER_AREA)

                # Normaliza a imagem, dividindo os valores dos pixels por 255.0
                normalized_img = cropped_img / 255.0
                
                # Adiciona a dimensão do batch para a entrada do modelo
                final_image = np.expand_dims(normalized_img, axis=0)
                
                # Realiza a predição da emoção
                prediction = model.predict(final_image, verbose=0)[0]
                emotion_label = emotion_labels[np.argmax(prediction)]
                
                # Adiciona o texto da emoção no frame
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            # Converte a imagem BGR para RGB para que o Streamlit a exiba corretamente
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Exibe o frame no placeholder
            frame_placeholder.image(rgb_frame, channels="RGB")

        # Libera a câmera quando a aplicação é interrompida
        cap.release()
        frame_placeholder.empty()
        