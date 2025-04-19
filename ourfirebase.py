from firebase import firebase
url="https://driverfinalv2-default-rtdb.firebaseio.com/"
firebase = firebase.FirebaseApplication(url)

def send_to_database(s):
        if(s == True):
# #             from firebase import firebase
# #             url="https://driverfinalv2-default-rtdb.firebaseio.com/"
# #             firebase = firebase.FirebaseApplication(url)
#             firebase.put("/Employees/5zZOEcaTbWc5qQKss9MY3UIh2y43", "drowsy",1)
            f = open("temp.txt", "a")
            f.write("1")
            firebase.put("/Employees/itwVefusLWh0R6ifPC3AANyOtQg1", "drowsy","1")
            f.close()
        else:
#             from firebase import firebase
#             url="https://driverfinalv2-default-rtdb.firebaseio.com/"
#             firebase = firebase.FirebaseApplication(url)
#             firebase.put("/Employees/5zZOEcaTbWc5qQKss9MY3UIh2y43", "drowsy",0)
            f = open("temp.txt", "a")
            f.write("0")
            firebase.put("/Employees/itwVefusLWh0R6ifPC3AANyOtQg1", "drowsy","0")
            f.close()