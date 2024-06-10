import cv2
import requests

class AlertGenerator:
    def generate_alert(self,alert, db_manager):
        db_manager.record_alert(alert)
        message = f"New Alert Generated : {alert.time}"
        self.update_flask_data(message)
        print(f"Alert: Recognized {alert.alert_type} {alert.description}")

    def update_flask_data(self,message):
        response = requests.post(f'http://127.0.0.1:5000/update-data/{message}')

        '''
        if response.status_code == 200:
            print("Data updated successfully")
        else:
            print("Failed to update data")
        '''

    #testing
    @staticmethod
    def generate_alert_test(name, face_type, frame):
        #file_name = name + " " + face_type + " "+ str(datetime.time.now()) + ".jpg"
        file_name = name + " " + face_type + ".jpg"
        print(file_name)
        success = cv2.imwrite(file_name, frame)
        if not success:
            print("Error Saving.....")

        print(f"Alert: Recognized {face_type} {name}")
