from sms_service import SMSService
import unittest
from unittest.mock import patch, MagicMock

class TestSMSService(unittest.TestCase):
    def setUp(self):
        self.sms = SMSService(api_key="test_key")

    @patch('requests.post')
    def test_send_sms_success(self, mock_post):

        mock_response = MagicMock()
        mock_response.json.return_value = {"return": True, "message": "Success"}
        mock_post.return_value = mock_response

        result = self.sms.send_sms("9876543210", "Test Message")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "SMS sent successfully")

    @patch('requests.post')
    def test_send_sms_failure(self, mock_post):
    
        mock_response = MagicMock()
        mock_response.json.return_value = {"return": False, "message": "Invalid API Key"}
        mock_post.return_value = mock_response

        result = self.sms.send_sms("9876543210", "Test Message")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Invalid API Key")

    def test_no_api_key(self):
        sms_no_key = SMSService(api_key="YOUR_FAST2SMS_API_KEY")
        result = sms_no_key.send_sms("9876543210", "Test")
        self.assertEqual(result["status"], "error")
        self.contains(result["message"], "API Key not configured")

    def contains(self, string, substring):
        self.assertTrue(substring in string)

if __name__ == '__main__':
    unittest.main()
