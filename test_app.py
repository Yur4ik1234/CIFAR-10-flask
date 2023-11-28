import os
import unittest
from app import app, allowed_file, predict
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.model = load_model('model.hdf5')

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_allowed_file(self):
        self.assertTrue(allowed_file("img1.jpeg"))
        self.assertTrue(allowed_file("img2.jpg"))
        self.assertTrue(allowed_file("img3.jpg"))

    def test_predict(self):
        image_path = './Testing-data/img2.jpg'  # Replace with an actual image path
        img = img_to_array(load_img(image_path, target_size=(32, 32)))
        img = img.reshape(1, 32, 32, 3)
        img = img.astype('float32')
        img = img / 255.0
        class_result, prob_result = predict(image_path, self.model)
        self.assertEqual(len(class_result), 8)
        self.assertEqual(len(prob_result), 8)

    def test_success_route(self):
        response = self.app.get('/success')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
