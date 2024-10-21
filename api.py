import requests 

url = "https://srv618269.hstgr.cloud/api/insurances"
response = requests.get(url)
data = response.json()
print(data)


