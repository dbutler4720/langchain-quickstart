import os
from aiohttp import request
from openai import OpenAI
import cv2
from huggingface_hub import snapshot_download
from transformers import VisionEncoder, TextModel
from PIL import Image
from time import time
from utils import ensure_dir_exists
import numpy as np
import requests


NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
CYAN = '\033[96m'

# Initialize the OpenAI client with the API key
client = OpenAI()

def mistral7b(user_input):
    streamed_completion = client.chat.completions.create(
        model = "davinci",
        messages=[
            {"role": "system", "content": "You are an expert at writing security logs."},
            {"role": "user", "content": user_input}
        ],
        stream=True, # Enable streaming
    )

    full_response = ""
    line_buffer = ""

    for chunk in streamed_completion:
        delta_content = chunk.choices[0].delta.content

        if delta_content is not None:
            line_buffer += delta_content

            if '\n' in line_buffer:
                lines = line_buffer.split('\n')
                for line in lines[:-1]:
                    print(NEON_GREEN + line + RESET_COLOR)
                    full_response += line + '\n'
                line_buffer = lines[-1]

    if line_buffer:
        print(NEON_GREEN + line_buffer + RESET_COLOR)
        full_response += line_buffer

    return full_response

def process_image(images_dir):
    # Find all images in the directory
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]

    if not images:
        print("No images found in the directory.")
        return None, None
    # Sort images by modification time (or you can sort by name if they are named with a timestamp
    latest_image = max(images, key=os.path.getmtime)

    # Remove all other images
    for image in images:
        if image!= latest_image:
            os.remove(image)
            print(f"Removed {image}")

    # Now, process the latest image
    model_path = snapshot_download("vikhyatk/moondream1")
    vision_encoder = VisionEncoder(model_path)
    text_model = TextModel(model_path)

    image = Image.open(latest_image)
    image_embeds = vision_encoder(image)
    return text_model, image_embeds

def start_video_capture(stream_url, output_dir='captured_video', capture_duration=5):
    """
    Captures video from the stream for a specified duration and saves it as .mp4 using H.264 codec
    Returns the path to the saved video file.
    
    :param stream_url: URL of the video stream.
    :param output_dir: Directory to save the captured video.
    :param capture_duration: Duration of the video capture in seconds.
    :return: Path to the saved video file.
    """
    ensure_dir_exists(output_dir)

    cap = cv2.VideoCapture(stream_url)

    # Attempt to set the capture to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Unable to open the video stream.")
        return None
    
    # Get actual video frame size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object
    # Attempt to use H.264 codec (X264' or 'avc1')
    # Note: You might need to change this based on your system
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    output_path = os.path.join(output_dir, f'captured_{int(time.time())}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    start_time = time.time()
    print(f"Capturing video for {capture_duration} seconds...")

    while int(time.time() - start_time) < capture_duration:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video.")
            break
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def send_mail(recipient, subject, body, attachment=None):
    # data = {
    #     "from": "dgb - <dgbutler4720@gmail.com>",
    #     "to": recipient,
    #     "subject": subject,
    #     "text": body,
    # }

    # if attachment is not None:
    #     with open(attachment, "rb") as f:
    #         files = {'attachment': (os.path.basename(attachment), f)}
    #         response = request.post(
    #             "https://api.mailgun.net/v3/sandbox5555555555555555",
    #             auth=("api", mailgun_api_key),
    #             data=data,
    #             files=files
    #         )
    # else:
    #     response = requests.post(
    #         "https://api.mailgun.net/v3/sandbox5555555555555555",
    #         auth=("api", mailgun_api_key),
    #         data=data
    #     )

    # if response.status_code!= 200:
    #     raise Exception(f"Unable to send email: " + str(response.text))
    
    print(f"Fake Email (not) sent to {recipient}")

def capture_frame(stream_url, images_dir='captured_images'):
    url = 'https://192.168.0.5:8080/shot.jpg'

    raw_data = requests.get(url, verify=False)
    image_array = np.array(bytearray(raw_data.content), dtype=np.uint8)
    frame = cv2.imdecode(image_array, -1)
    output_path = os.path.join(images_dir, f'captured_{int(time.time())}.jpg')
    cv2.imwrite(output_path, frame)
    print('saving frame...')

def main():
    stream_url = 'http://192.168.0.5:8080/video'
    images_dir = './images'
    video_dir = './videos'

    person_detected = False
    while True:
        capture_frame(stream_url, images_dir=images_dir)
        text_model, image_embeds = process_image(images_dir)
        prompt = f"{NEON_GREEN}Is there a PERSON in the image?{RESET_COLOR} (ONLY ANSWER YES OR NO)"

        print (">", prompt)

        # Ensure text_model and image_embeds are valid before proceeding
        if text_model is not None and image_embeds is not None:
            answer = text_model.answer_question(image_embeds, prompt).strip().upper()
        else:
            print("Could not process image properly.")
            continue

        print(CYAN + answer + RESET_COLOR)

        if answer == "YES":
            person_detected = True
            # Start video capture if a person is detected
            print(f"{NEON_GREEN}Person detected! Starting video capture...{RESET_COLOR}")
            video_path = start_video_capture(stream_url, output_dir=video_dir, capture_duration=5)
            print("Sending email...")
            recipient = "dgbutler4720@gmail.com"
            subject = "Video Clip"
            body = "Here is a 5 second video clip"
            send_mail(recipient, subject, body, attachment=video_path)
            text_model, image_embeds = process_image(images_dir)
            prompt2 = "Describe the image in detail, try to identify Gender, Objects, Clothing, etc:"
            answer2 = text_model.answer_question(image_embeds, prompt2)
            log = f"Image Description: {answer2} \n From the description write a security log:"
            log2 = mistral7b(log)
            # save_file("Security_log.txt", log2)
            print(f"Security_log: {log2}")
            # You might want to break or continue here depending on your application
            # break
        elif answer == "NO":
            person_detected = False
        else:
            print("Invalid answer. Please respond with 'YES' or 'NO'.")
            continue

        # Output the status of the person detection
        print(f"Person detected: {person_detected}")

        # If you want to break the loop after getting a valid response, uncomment the following line
        # break

if __name__ == "__main__":
    main()