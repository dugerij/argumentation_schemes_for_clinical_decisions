import os
from ollama import chat
from dotenv import load_dotenv

from utils import startup_check

load_dotenv()
startup_check() # Ensure required variables are set

response = chat(
    model=os.environ.get('GENERATOR_BASE_MODEL'),
    messages=[
        {
            'role': 'user',
            'content':  'A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?'
            }
        ],
    think=False,
)

print(response.message.content)