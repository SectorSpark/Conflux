import io
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import time

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2' #Теперь загрузка приложения должны быть до обработки фото
hub_module = hub.load(hub_handle) 
  
def check_size(shape):
  if (shape[1] >= 256) and (shape[2] >= 256) and (shape[1] < 4000) and (shape[2] < 4000):
    return True
  else:
    st.write('Что-то пошло не так. Стороны изображения должны быть больше 256 и меньше 4000.')
    return False
      
def crop_center(image):
  shape = image.shape
  if check_size(shape) == True:
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image
 
def preprocess_image(img, resize):
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, resize, preserve_aspect_ratio=True)
    return img

def load_image(imageLabel):
  try:
    uploaded_file = st.file_uploader(label=imageLabel)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return image_data
    else:
        return None
  except:
     st.write('Что-то пошло не так. Проверьте загружаемый файл.')

st.title('Стилизация изображений')
col1, col2 = st.columns(2)
with col1:
  img = load_image('Первое изображение')    
with col2:
  img2 = load_image('Второе изображение') 
result = st.button('Применить стиль второго изображения к первому')

if result:
    content_image = preprocess_image(img,(1024,1024)) # Увеличена скоростть обработки за счет понижения колличества пикселей. Изменение разрешения не повлияло на обрезку фото
    style_image = preprocess_image(img2,(256,256))
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    st.write('**Результат:**')
    with st.spinner("Идёт обработка..."): #Добалена шкала загрузки
         time.sleep(5)
    st.success('Успешно!')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = np.squeeze(outputs[0])
  
    st.image(stylized_image)
