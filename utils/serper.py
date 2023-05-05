import requests
import http.client
import json

def getSerperResults(conn, query, SERPER_API_KEY):
    payload = json.dumps({ "q": query, "num": 1 })
    headers = {
    'X-API-KEY': SERPER_API_KEY,
    'Content-Type': 'application/json'
    }
    
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")