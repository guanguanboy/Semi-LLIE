import requests
import os
import samscore

def download_image(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    with open(save_path, 'wb') as file:
        file.write(response.content)
os.makedirs('imgs', exist_ok=True)
# Example usage
image_url = 'https://i.ibb.co/yFFg5pn/n02381460-20-real.png'
save_location = 'imgs/real.png'
download_image(image_url, save_location)

image_url = 'https://i.ibb.co/GCQ2jQy/n02381460-20-fake.png'
save_location = 'imgs/fake.png'
download_image(image_url, save_location)

## Initializing the model
# You can choose "vit_t", "vit_l", "vit_b", "vit_h"
SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_b" )
samscore_result = SAMScore_Evaluation.evaluation_from_path(source_image_path='imgs/real.png',  generated_image_path='imgs/fake.png')

print('SAMScore: %.4f'%samscore_result)