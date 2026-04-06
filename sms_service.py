import requests
import os

class SMSService:
    def __init__(self, api_key=None):
        # Default placeholder API key - user should replace this
        self.api_key = api_key or os.environ.get("FAST2SMS_API_KEY")
        self.url = "https://www.fast2sms.com/dev/bulkV2"
        self.factor_url = "https://2factor.in/API/V1/{api_key}/SMS/{number}/{message}"

    def send_sms(self, number, message):
        """
        Sends an SMS using Fast2SMS API (Free Tier / Bulk V2).
        Requires a valid India phone number.
        """
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            print(f"\n≡ƒôó [MOCK SMS] To: {number}\n≡ƒô¥ Message: {message}\n")
            return {"status": "success", "message": "[MOCK] Sent successfully to terminal"}

        # Try 2Factor if the key looks like a 2Factor key (long hash)
        if "-" in self.api_key and len(self.api_key) > 30:
            try:
                final_url = self.factor_url.format(api_key=self.api_key, number=number, message=message)
                response = requests.get(final_url)
                return {"status": "success", "message": "SMS sent via 2Factor"}
            except Exception as e:
                return {"status": "error", "message": f"2Factor Error: {str(e)}"}

        # Default to Fast2SMS
        payload = {
            "message": message,
            "language": "english",
            "route": "q",
            "numbers": number,
        }
        
        headers = {
            "authorization": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response_data = response.json()
            
            if response_data.get("return"):
                return {"status": "success", "message": "SMS sent successfully"}
            else:
                return {"status": "error", "message": response_data.get("message", "Unknown error")}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Global instance
sms_handler = SMSService()
