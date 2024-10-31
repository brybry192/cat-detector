import unittest
from io import BytesIO
import os

# Import Flask app from script.
from cat_breed_detector import app

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_detect_cat_with_tabby_image(self):
        # Path to the image that should detect a tabby cat
        image_path = './images/beastie.jpg'
        self.assertTrue(os.path.exists(image_path), f"Image not found at {image_path}")

        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Send POST request with image file
        response = self.app.post('/detect_cat', data={
            'file': (BytesIO(image_bytes), 'beastie.jpg')
        }, content_type='multipart/form-data')

        # Parse response
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        # Assertions
        self.assertTrue(json_data['is_tabby'], "Expected to detect a tabby cat.")
        self.assertGreaterEqual(json_data['is_tabby_confidence'], 0.5)
        print('Test with tabby image response:', json_data)

    def test_detect_cat_with_non_cat_image(self):
        # Path to the image that should not detect a cat
        image_path = './images/nature_IMG_9026.jpg'
        self.assertTrue(os.path.exists(image_path), f"Image not found at {image_path}")

        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Send POST request with image file
        response = self.app.post('/detect_cat', data={
            'file': (BytesIO(image_bytes), 'nature_IMG_9026.jpg')
        }, content_type='multipart/form-data')

        # Parse response
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        # Assertions
        self.assertFalse(json_data['is_tabby'], "Expected not to detect a tabby cat.")
        self.assertGreater(json_data['is_tabby_confidence'], 0.8)
        self.assertEqual(json_data['breed'], 'No_Cat', "Expected breed to be 'No_Cat'.")
        print('Test with non-cat image response:', json_data)

if __name__ == '__main__':
    unittest.main()

