import pymysql
import numpy as np
from io import BytesIO
from PIL import Image
import face_recognition
from app.models.alert import Alert
import logging

class DatabaseManager:
    def __init__(self):
        self.db_host = "127.0.0.1"
        self.db_user = "root"
        self.db_password = ""
        self.db_name = "db_security_system"

    def create_conection(self):
        try:
            conn = pymysql.connect(host= self.db_host, user = self.db_user, password= self.db_password, database= self.db_name) 
            return conn
        except Exception as e:
            logging.info(str(e))

    def load_known_faces(self):

        conn = self.create_conection()
        if not conn:
            return None, None, None

        known_faces = []
        known_names = []
        face_types = []
        
        try:
            with conn.cursor() as cursor:
                query = "SELECT cif_no, photo FROM tbl_blacklisted"
                cursor.execute(query)
                rows = cursor.fetchall()
                
                face_type = "blacklisted"
                for row in rows:
                    name, image_data = row
                    image = Image.open(BytesIO(image_data))
                    image = np.array(image)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces.append(encoding)
                    known_names.append(name)
                    face_types.append(face_type)

                query = "SELECT robber_id, photo FROM tbl_robbers"
                cursor.execute(query)
                rows = cursor.fetchall()
                
                face_type = "robber"
                
                for row in rows:
                    name, image_data = row
                    image = Image.open(BytesIO(image_data))
                    image = np.array(image)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces.append(encoding)
                    known_names.append(name)
                    face_types.append(face_type)

        except Exception as e:
            logging.info(f"Error loading known faces: {e}")
        
        conn.close()
        
        return known_faces, known_names, face_types
        
    def get_last_alert(self):
        conn = self.create_conection()
        if not conn:
            return
        
        try:
            query = "SELECT type, description, branch_id, time, checked_by FROM tbl_alerts ORDER BY alert_id DESC"
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            cursor.close()
            conn.close()


            alert = Alert(
                alert_type=row[0],
                description=row[1],
                branch_id=row[2],
                time=row[3],
                photo=None,
                is_checked=row[4]
            )

            return alert

        except Exception as e:
            logging.info(str(e))
            
    def record_alert(self, alert):
        conn = self.create_conection()
        if not conn:
            return
        
        try:
            query = "INSERT INTO tbl_alerts (type, description, photo, time, branch_id, generated_by) VALUES (%s, %s, %s, %s, %s, %s)"
            data = (alert.alert_type, alert.description, alert.photo, alert.time, alert.branch_id,alert.generated_by,)
            cursor = conn.cursor()
            cursor.execute(query, data)
            conn.commit()
            conn.close()

        except Exception as e:
            logging.info(str(e))



