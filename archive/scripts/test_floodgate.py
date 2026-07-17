import requests
import os

os.environ['HTTPS_PROXY'] = 'http://proxy-aws-5-bolt.corp.apple.com:8080'
os.environ['HTTP_PROXY'] = 'http://proxy-aws-5-bolt.corp.apple.com:8080'

URL = "https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models/gemini-3.1-pro-preview:generateContent"
payload = {"contents": [{"role": "user", "parts": [{"text": "Hello, please explain the theory of relativity in one sentence."}]}]}
CERT_PATH = ('/turibolt_k8s_mounts/narrative/turi/cert.pem', '/turibolt_k8s_mounts/narrative/turi/private.pem')

try:
    r = requests.post(URL, json=payload, cert=CERT_PATH, timeout=60)
    print("Status:", r.status_code)
    if r.status_code == 200:
        print("Response:", r.json()['candidates'][0]['content']['parts'][0]['text'])
    else:
        print("Error:", r.text)
except Exception as e:
    print("Exception:", e)
