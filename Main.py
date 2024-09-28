# imports 
from openai import OpenAI
import requests

######################################################################
# Main 
######################################################################

def main():
    
    print("Testing image to text gen")
    
    client = OpenAI()
    
    pass

#####################################################################
# Extension Functions 
#####################################################################

def callToOpenAI(client, content, api_key):
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": content
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    x = requests.post(url, headers = headers, json = request)

    print(x.text)
    
    pass

if __name__ == "__main__":
    main()