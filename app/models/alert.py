#from ...config import Config

class Alert:
    def __init__(self,alert_type, photo, time, description,branch_id = "179", is_checked=False):
        self.alert_type = alert_type
        self.description = description
        self.photo = photo
        self.time = time
        self.branch_id = branch_id
        self.generated_by = "AUTO"
        self.is_checked = is_checked
        