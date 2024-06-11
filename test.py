#Test App

from app.models.database_manager import DatabaseManager
from app.models.alert_generator import AlertGenerator

def test():
    
    db_manager = DatabaseManager()

    alert = db_manager.get_last_alert()

    message = f'Type : {alert.alert_type}\nDesc : {alert.description}\nbranch : {alert.branch_id}\ntime : {alert.time}'
    #message = str(alert)
    #print(message)

    alert_genarator = AlertGenerator()
    