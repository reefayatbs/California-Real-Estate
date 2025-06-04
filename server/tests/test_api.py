import unittest
import json
from server import app
import util

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        util.load_artifacts()

    def test_get_cities(self):
        response = self.app.get('/api/v1/cities')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['cities'], list)
        self.assertTrue(len(data['cities']) > 0)

    def test_predict_valid_input(self):
        test_data = {
            'house_size': 2000,
            'city': 'Los Angeles',
            'bed': 3,
            'bath': 2
        }
        response = self.app.post('/api/v1/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['estimated_price'], (int, float))

    def test_predict_invalid_input(self):
        test_data = {
            'house_size': -100,  # Invalid negative value
            'city': 'Los Angeles',
            'bed': 3,
            'bath': 2
        }
        response = self.app.post('/api/v1/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertIn('errors', data)

    def test_predict_missing_fields(self):
        test_data = {
            'house_size': 2000,
            'city': 'Los Angeles'
            # Missing bed and bath
        }
        response = self.app.post('/api/v1/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertIn('errors', data)

if __name__ == '__main__':
    unittest.main() 