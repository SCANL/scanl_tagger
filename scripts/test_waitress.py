import threading
import requests

host = input('hostname? [0.0.0.0]: ')
port = input('port? [5000]: ')

def req(word,type,id):
    if host == '' or port == '':
        print(str(id) + ": " + requests.get(url = f'http://127.0.0.1:5000/{word}/{type}', params = {}).text)
    else:
        print(str(id) + ": " + requests.get(url = f'http://{host}:{port}/{word}/{type}', params = {}).text)

r1 = threading.Thread(target=req, args=("numberArray","DECLARATION",1))
r2 = threading.Thread(target=req, args=("GetNumberArray","FUNCTION",2))
r3 = threading.Thread(target=req, args=("PersonRecord","CLASS",3))

r1.start()
r2.start()
r3.start()

r1.join()
r2.join()
r3.join()