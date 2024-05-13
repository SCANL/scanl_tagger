import threading
import requests

host = input('hostname? [0.0.0.0]: ')
port = input('port? [5000]: ')

def req_one():
    if host == '' or port == '': 
        print(requests.get(url = 'http://127.0.0.1:5000/numberArray/DECLARATION', params = {}))
    else: print(requests.get(url = 'https://'+host+':'+port+'/numberArray/DECLARATION', params = {}))

def req_two():
    if host == '' or port == '': 
        print(requests.get(url = 'http://127.0.0.1:5000/GetNumberArray/FUNCTION', params = {}))
    else: print(requests.get(url = 'https://'+host+':'+port+'/GetNumberArray/FUNCTION', params = {}))
    
def req_three():
    if host == '' or port == '': 
        requests.get(url = 'http://127.0.0.1:5000/PersonRecord/CLASS', params = {})
    else: print(requests.get(url = 'https://'+host+':'+port+'/PersonRecord/CLASS', params = {}))

r1 = threading.Thread(target=req_one, args=())
r2 = threading.Thread(target=req_two, args=())
r3 = threading.Thread(target=req_three, args=())

r1.start()
r2.start()
r3.start()

r1.join()
r2.join()
r3.join()
