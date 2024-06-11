import cv2
import requests
from app.models.database_manager import DatabaseManager

class AlertGenerator:
    def __init__(self):
        db_manager = DatabaseManager()
        self.last_alert = db_manager.get_last_alert()

    
    def set_last_alert(self):
        db_manager = DatabaseManager()
        self.last_alert = db_manager.get_last_alert()

    def is_new_alert(self, alert):
        if alert.alert_type == self.last_alert.alert_type and alert.description == self.last_alert.description and alert.branch_id == self.last_alert.branch_id:
            return False
        
        return True
        
    def generate_alert(self,alert, db_manager):
        if self.is_new_alert(alert):
            db_manager.record_alert(alert)
            self.set_last_alert()
            message = f"New Alert Generated : {alert.time}"
            try:
                self.update_flask_data(message)
            except Exception as e:
                print(str(e))

        print(f"Alert: Recognized {alert.alert_type} {alert.description} {alert.time}")

    def update_flask_data(self,message):
        response = requests.post(f'http://127.0.0.1:5000/update-data/{message}')

        '''
        if response.status_code == 200:
            print("Data updated successfully")
        else:
            print("Failed to update data")
        '''

