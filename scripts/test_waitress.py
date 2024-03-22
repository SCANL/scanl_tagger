import threading
import requests

def req_one():
    requests.get(url = 'http://127.0.0.1:5000/numberArray/DECLARATION', params = {})
    
def req_two():
    requests.get(url = 'http://127.0.0.1:5000/GetNumberArray/FUNCTION', params = {})
    
def req_three():
    requests.get(url = 'http://127.0.0.1:5000/PersonRecord/CLASS', params = {})

r1 = threading.Thread(target=req_one, args=())
r2 = threading.Thread(target=req_two, args=())
r3 = threading.Thread(target=req_three, args=())

r1.start()
r2.start()
r3.start()

r1.join()
r2.join()
r3.join()
