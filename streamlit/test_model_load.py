
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = 'mobilenetv2_finetuned_fer2013_v2.keras'

print("Tentando carregar o modelo...")
try:
    model = load_model(model_path)
    print("Modelo carregado com sucesso!")
    model.summary()
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Causas possíveis:")
    print("1. O arquivo não está na pasta correta.")
    print("2. O nome do arquivo está incorreto.")
    print("3. O arquivo está corrompido.")
    print("4. Versão incompatível do TensorFlow/Keras.")