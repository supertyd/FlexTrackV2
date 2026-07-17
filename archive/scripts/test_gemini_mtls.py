import requests

CERT_PATH = '/turibolt_k8s_mounts/narrative/turi/chain.pem'
KEY_PATH = '/turibolt_k8s_mounts/narrative/turi/private.pem'

url = 'https://floodgate.g.apple.com/api/gemini/v1/publishers/google/models/gemini-2.5-pro:generateContent'

try:
    response = requests.post(url,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'FlexTrackV2-Evaluation/1.0'
        },
        cert=(CERT_PATH, KEY_PATH),
        json={
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello, Gemini! Please respond with a single sentence."}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 128
            }
        }
    )
    if response.status_code == 200:
        result = response.json()
        print('SUCCESS!')
        print(result['candidates'][0]['content']['parts'][0]['text'])
    else:
        print(f'FAILED: {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'ERROR: {e}')
