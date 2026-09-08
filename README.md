# Produce Freshness Scanner
This is a computer vision program that analyzes produce through images and returns freshness status.
## Background
What motivated me to create this application was my experience in the produce section of H-E-B. Because I'm a computer science major and also happen to work in produce, I thought about what project would help advance both produce and software engineering, so this is where the Produce Freshness Scanner idea was born. I decided to make this program similar to the barcode scanners attached to phones across the store when it came to quantity tracking, except that this application uses images. This way, it could help prevent shortages or help young people especially children train their intuition when deciding which fruit or vegetable is rotten.
## Technologies used and why
- Python - This is the most common programming language used in machine learning.
- PyTorch - To enable MobileNetV3 for computer vision across images.
- SafeTensors - A safer alternative to pickle files or code for proper security.
- FastAPI - To allow asynchronous API endpoints.
- Streamlit - To write the front-end in a simple manner using Python.
- Render - To allow the backend along with UptimeRobot and the Dockerfile to stay awake 24/7.
## How to use the application
<img width="1919" height="942" alt="image" src="https://github.com/user-attachments/assets/c28e8840-2f26-4ad5-93fc-b298ec246eef" />

Click on the Streamlit link on the right side of the screen below the About section and description.
 
In case the app says "This app has gone to sleep due to inactivity. Would you like to wake it back up?", simply press "Yes, get this app back up!" and you should be directed to a page that looks like this. <img width="1919" height="941" alt="image" src="https://github.com/user-attachments/assets/39247960-d4e4-4a07-8a60-2d97d5ba0401" />

Afterwards, choose either a file upload to upload photos or choose camera capture to take a live photo. 

Then press "Analyze Freshness", and you will have results of either "FRESH" or "ROTTEN" along with the model confidence.
## Performance
<img width="1089" height="450" alt="image" src="https://github.com/user-attachments/assets/c3c19f38-e87c-482c-81dc-df70315000b7" />

## Limitations
False positives and negatives are possible, so this application should not be used for quality assurance and should be instead used experimentally. For more precise results, photos should be taken close to a product and with a simpler background.
## License
This project is licensed under the [MIT License](LICENSE).
