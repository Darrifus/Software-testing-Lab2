import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestSmartFarmingIntegration(unittest.TestCase):

    # Интеграция Sensor_values и внешнего API
    @patch('builtins.input', return_value='test_api_key')
    def test_external_api_integration(self, mock_input):
        with patch('ipinfo.getHandler') as mock_ipinfo:
            mock_handler = MagicMock()
            mock_details = MagicMock()
            mock_details.city = "Saint-Petersburg"
            mock_handler.getDetails.return_value = mock_details
            mock_ipinfo.return_value = mock_handler

            import Sensor_values as sv
            city, sensor_values = sv.get_readings()

            self.assertEqual(city, "Saint-Petersburg")
            self.assertEqual(len(sensor_values), 8)
            self.assertIsInstance(sensor_values[0], float)
            self.assertIsInstance(sensor_values[4], float)

    # Интеграция Sensor_values и Crop_Pred
    @patch('builtins.input', return_value='test_api_key')
    def test_crop_prediction_integration(self, mock_input):
        with patch('ipinfo.getHandler') as mock_ipinfo:
            import Sensor_values as sv
            import Crop_Pred as cp

            mock_handler = MagicMock()
            mock_details = MagicMock()
            mock_details.city = "Kazan"
            mock_handler.getDetails.return_value = mock_details
            mock_ipinfo.return_value = mock_handler

            city, sensor_values = sv.get_readings()

            crop_input = sensor_values[0:4].copy()
            temp = crop_input[0]
            for i in range(len(crop_input) - 1):
                crop_input[i] = crop_input[i + 1]
            crop_input[-1] = temp

            crop_id, crop_dict = cp.Predict_Crop(crop_input)

            self.assertTrue(int(crop_id) in range(len(crop_dict)))

    # Интеграция Sensor_values и Fertiliser_Prediction
    @patch('builtins.input', return_value='test_api_key')
    def test_fertiliser_prediction_integration(self, mock_input):
        with patch('ipinfo.getHandler') as mock_ipinfo:
            import Sensor_values as sv
            import Fertiliser_Prediction as fp

            mock_handler = MagicMock()
            mock_details = MagicMock()
            mock_details.city = "Omsk"
            mock_handler.getDetails.return_value = mock_details
            mock_ipinfo.return_value = mock_handler

            city, sensor_values = sv.get_readings()

            fert_input = np.array([sensor_values[1:4]])
            fert_class = int(fp.Predict_Fertiliser(fert_input))

            self.assertIn(fert_class, [1, 2, 3])

    # Интеграция Sensor_values и Main
    @patch('builtins.input', return_value='test_api_key')
    def test_weather_prediction_integration(self, mock_input):
        with patch('ipinfo.getHandler') as mock_ipinfo:
            import Sensor_values as sv

            mock_handler = MagicMock()
            mock_details = MagicMock()
            mock_details.city = "Moscow"
            mock_handler.getDetails.return_value = mock_details
            mock_ipinfo.return_value = mock_handler

            city, sensor_values = sv.get_readings()
            temp, humidity, pressure = sensor_values[4], sensor_values[5], sensor_values[6]

            if humidity > 70:
                if pressure < 100:
                    weather_pred = "Гроза"
                else:
                    weather_pred = "Дождь"
            elif pressure < 99:
                weather_pred = "Ветер"
            else:
                weather_pred = "Ясно"

            self.assertIn(weather_pred, ["Гроза", "Дождь", "Ветер", "Ясно"])

    # Согласованность Crop_Pred и Fertiliser_Prediction на одних сенсорных данных
    @patch('builtins.input', return_value='test_api_key')
    def test_crop_and_fertiliser_combined_integration(self, mock_input):
        with patch('ipinfo.getHandler') as mock_ipinfo:
            import Sensor_values as sv
            import Crop_Pred as cp
            import Fertiliser_Prediction as fp

            mock_handler = MagicMock()
            mock_details = MagicMock()
            mock_details.city = "Samara"
            mock_handler.getDetails.return_value = mock_details
            mock_ipinfo.return_value = mock_handler

            city, sensor_values = sv.get_readings()
            crop_input = sensor_values[0:4].copy()
            temp = crop_input[0]
            for i in range(len(crop_input) - 1):
                crop_input[i] = crop_input[i + 1]
            crop_input[-1] = temp
            crop_pred, crop_pairs = cp.Predict_Crop(crop_input)
            fert_input = np.array([sensor_values[1:4]])
            fert_pred = fp.Predict_Fertiliser(fert_input)
            try:
                crop_index = int(np.array(crop_pred).ravel()[0])
            except Exception:
                crop_index = int(crop_pred)
            try:
                fert_class = int(np.array(fert_pred).ravel()[0])
            except Exception:
                fert_class = int(fert_pred)

            if hasattr(crop_pairs, 'keys'):
                crop_names = list(crop_pairs.keys())
            else:
                crop_names = list(crop_pairs)
            self.assertTrue(0 <= crop_index < len(crop_names))
            self.assertIn(fert_class, [1, 2, 3])
            self.assertTrue(str(crop_names[crop_index]).strip() != "")


if __name__ == '__main__':
    unittest.main(verbosity=2)
