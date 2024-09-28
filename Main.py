# imports 
from openai import OpenAI
import requests

######################################################################
# Main 
######################################################################

def main():
    
    print("Testing image to text gen")
    
    client = OpenAI()
    callToOpenAI("", "What is this image showing in one word?", "", "")
    
    pass

#####################################################################
# Extension Functions 
#####################################################################

def callToOpenAI(client, content, img, api_key):
    
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
                "type": "image_file",
                "image_file": {
                    "file_id": img,
                    "purpose" : "vision"
                    }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post(url, headers = headers, json = request)

    print(response.json)
    
    pass

if __name__ == "__main__":
    main()