import tensorflow as tf
import gradio as gr
import numpy as np

# Load model
model = tf.keras.models.load_model("waste_classifier.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Prediction function
def predict(img):
    img = tf.image.resize(img, [224, 224])
    img = tf.expand_dims(img, axis=0)
    img = img / 255.0
    preds = model.predict(img)[0]
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

# Gradio app
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="♻️ Garbage Classification",
    description="Upload a waste image and get its type (Metal, Paper, Plastic etc.)."
)

demo.launch()
