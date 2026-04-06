import requests
import os

token = "6a677953b0a18a7c092a419d0e194535c926caa1"
uri = "https://dagshub.com/krishnabhagavan910/generic-mlops-pipeline.mlflow/api/2.0/mlflow/experiments/list"

# Method 1: Token as username, empty password
print("Method 1:")
r1 = requests.get(uri, auth=(token, ""))
print(r1.status_code)

# Method 2: Username and token as password
username = "krishnabhagavan910"
print("\nMethod 2:")
r2 = requests.get(uri, auth=(username, token))
print(r2.status_code)

# Method 3: Token as token
print("\nMethod 3 (Bearer):")
r3 = requests.get(uri, headers={"Authorization": f"Bearer {token}"})
print(r3.status_code)

# Method 4: Token as Basic Auth username and "x-oauth-basic" as password (GitHub style)
print("\nMethod 4 (Token:x-oauth-basic):")
r4 = requests.get(uri, auth=(token, "x-oauth-basic"))
print(r4.status_code)
