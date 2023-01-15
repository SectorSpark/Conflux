import image_classification as m
import tensorflow as tf
import pytest
import os
import numpy as np

def load_image_tf(image_url):
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  return img
    
#Проверка размера изображения. Не меньше 256х256
def test_check_size_small():
    image = load_image_tf('https://townandcountryremovals.com/wp-content/uploads/2013/10/firefox-logo-200x200.png')
    shape = image.shape
    assert m.check_size(shape) == False
    
#Проверка размера изображения. Не больше 4000х4000
def test_check_size_big():
    image = load_image_tf('https://s1.1zoom.ru/big3/416/Earth_Black_background_548690_4000x4000.jpg')
    shape = image.shape
    assert m.check_size(shape) == False
