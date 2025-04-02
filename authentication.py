import pyrebase

config = {
  'apiKey': "AIzaSyBoUCKaswlxAlXTyO_5LCDjl10lEXqKmNg",
  'authDomain': "evanfypworking.firebaseapp.com",
  'projectId': "evanfypworking",
  'storageBucket': "evanfypworking.firebasestorage.app",
  'messagingSenderId': "413702763958",
  'appId': "1:413702763958:web:e4411b617ab80442f3bd17",
  'measurementId': "G-SLN4E8LJYN",
  'databaseURL': "https://evanfypworking-default-rtdb.firebaseio.com/",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

email = 'test@gmail.com'
password = '123456'

#user = auth.create_user_with_email_and_password(email, password)
#print(user)

user = auth.sign_in_with_email_and_password(email, password)

#info = auth.get_account_info(user['idToken'])
#print(info)

auth.send_email_verification(user['idToken'])

auth.send_password_reset_email(email)
