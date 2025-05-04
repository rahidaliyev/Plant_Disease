from tensorflow.keras.models import load_model
from plantvillage import predict_image

model = load_model("plant_disease_model.h5")

result = predict_image("test_image.jpg")
print(f"Disease: {result}")