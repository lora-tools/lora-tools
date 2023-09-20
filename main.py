import os
import re
import shutil
from dotenv import load_dotenv
from typing import Any, List, Optional, Union, Dict
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gradio_client import Client
import httpx
import secrets
import requests
from pyngrok import ngrok, conf
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import bcrypt
import base64
import asyncio
import platform
import psutil
import hashlib
try:
    import GPUtil
except ImportError:
    GPUtil = None

BEARER_TOKENS = {}

# Load environment variables from .env file
load_dotenv()

# Get environment variables
KOHYA_URL = os.getenv("KOHYA_URL")
BASE_TRAINING_DIR = os.getenv("BASE_TRAINING_DIR")
NGROK_AUTH_TOKEN = os.getenv("NGROKAUTHTOKEN")
NGROK_PORT = os.getenv("NGROK_PORT")
TRAINING_COMPLETE_URL = os.getenv("TRAINING_COMPLETE_URL")
LORA_PATH = os.getenv("LORA_PATH")
PROXY_FETCH_URL = os.getenv("PROXY_FETCH_URL")
SDAPIURL = os.getenv("SDAPIURL")
SDAPIUSERNAME = os.getenv("SDAPIUSERNAME")
SDAPIPASSWORD = os.getenv("SDAPIPASSWORD")
SERVERID = os.getenv("SERVERID")
SERVER_PING_URL = os.getenv("SERVER_PING_URL")
BASE_PATH = os.getenv("BASE_PATH")

# Create a FastAPI instance
app = FastAPI()

# Set up HTTP Basic authentication 
security = HTTPBasic()

PROXY_API_URL = None  # Initialize it to None


async def fetch_proxy_api_url():
    """
    Fetch the PROXY_API_URL from the specified endpoint.
    """
    data = {"server": SERVERID}
    async with httpx.AsyncClient() as client:
        response = await client.post(PROXY_FETCH_URL, json=data)
        response.raise_for_status()
        return response.json()["response"]["url"]


async def ping_server():
    """Function to send a POST request to SERVER_PING_URL every 60 mins."""
    while True:
        try:
            response = httpx.post(SERVER_PING_URL, json={"server": SERVERID})
            if response.status_code != 200:
                print(f"Error pinging server: {response.text}")
        except Exception as e:
            print(f"Exception during ping: {str(e)}")
        
        await asyncio.sleep(3600)  # Wait for 3600 seconds (1 hr) before sending the next ping



# A function to send requests with the authorization header
async def fetch_valid_credentials(credentials: HTTPBasicCredentials):
    # Create the authentication header using the provided credentials
    headers = create_basic_auth_header(credentials.username, credentials.password)
    
    response = requests.get(f"{PROXY_API_URL}/get-credentials", headers=headers)
    response.raise_for_status()
    return response.json()

async def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    valid_credential = await fetch_valid_credentials(credentials)
    
    correct_username = secrets.compare_digest(credentials.username, valid_credential["username"])
    stored_password_hash = valid_credential["password"].encode('utf-8')
    password_verified = bcrypt.checkpw(credentials.password.encode('utf-8'), stored_password_hash)
    
    if correct_username and password_verified:
        return credentials
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )

def create_basic_auth_header(username: str, password: str) -> dict:
    """Create a header dictionary for basic authentication."""
    credentials = f"{username}:{password}"
    base64_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {base64_credentials}"}


class UserCredentials:
    def __init__(self, username, password):
        self.username = username
        self.password = password

# Pydantic models for request bodies
class LoRaData(BaseModel):
    username: str
    output_name: str

class TrainingData(BaseModel):
    username: str
    triggerword: str
    classword: str
    repeats: str
    output_name: str
    prompt: str
    # imageUrls: List[str]
    image_data: List[dict]  # change from imageUrls to image_data

class MonitorInput(BaseModel):
    webhook_url: str

# Function to convert text to leet speak
def to_leet_speak(text):
    leet_dict = {
        'a': '4',
        'e': '3',
        'i': '1',
        'o': '0',
        'u': '2',
        'A': '4',
        'E': '3',
        'I': '1',
        'O': '0',
        'U': '2'
    }
    pattern = re.compile("|".join(leet_dict.keys()))
    return pattern.sub(lambda m: leet_dict[m.group(0)], text)

# Pydantic model for TrainModelInput request body
class TrainModelInput(BaseModel):
    filepath_to_json: Optional[str]
    parameter_217_label_component: str
    pretrained_model_name_or_path: str
    v2: bool
    v_parameterization: bool
    sdxl_model: bool
    logging_folder: str
    image_folder: str
    regularisation_folder: str
    output_folder: str
    max_resolution: str
    learning_rate: float
    lr_scheduler: str
    lr_warmup_percentage_of_steps: int
    train_batch_size: int
    epoch: int
    save_every_n_epochs: int
    mixed_precision: str
    save_precision: str
    seed: str
    number_of_cpu_threads_per_core: int
    cache_latents: bool
    cache_latents_to_disk: bool
    caption_extension: str
    enable_buckets: bool
    gradient_checkpointing: bool
    full_fp16_training_experimental: bool
    no_token_padding: bool
    stop_text_encoder_training: int
    minimum_bucket_resolution: int
    maximum_bucket_resolution: int
    use_xformers: bool
    save_trained_model_as: str
    shuffle_caption: bool
    save_training_state: bool
    resume_from_saved_training_state: str
    prior_loss_weight: float
    text_encoder_learning_rate: float
    unet_learning_rate: float
    network_rank_dimension: int
    lora_network_weights: str
    dim_from_weights: bool
    color_augmentation: bool
    flip_augmentation: bool
    clip_skip: int
    gradient_accumulate_steps: int
    memory_efficient_attention: bool
    model_output_name: str
    model_quick_pick: str
    max_token_length: str
    max_train_epoch: str
    max_num_workers_for_data_loader: str
    network_alpha: int
    training_comment: str
    keep_n_tokens: int
    lr_number_of_cycles: str
    lr_power: str
    persistent_data_loader: bool
    dont_upscale_bucket_resolution: bool
    random_crop_instead_of_center_crop: bool
    bucket_resolution_steps: int
    dropout_caption_every_n_epochs: int
    rate_of_caption_dropout: int
    optimizer: str
    optimizer_extra_arguments: str
    noise_offset_type: str
    noise_offset: float
    adaptive_noise_scale: float
    multires_noise_iterations: float
    multires_noise_discount: float
    lora_type: str
    lokr_factor: float
    use_cp_decomposition: bool
    lokr_decompose_both: bool
    ia3_train_on_input: bool
    convolution_rank_dimension: int
    convolution_alpha: int
    sample_every_n_steps: int
    sample_every_n_epochs: int
    sample_sampler: str
    sample_prompts: str
    additional_parameters: str
    vae_batch_size: int
    min_snr_gamma: int
    down_lr_weights: str
    mid_lr_weights: str
    up_lr_weights: str
    blocks_lr_zero_threshold: str
    block_dims: str
    block_alphas: str
    conv_dims: str
    conv_alphas: str
    weighted_captions: bool
    dylora_unit_block_size: int
    save_every_n_steps: int
    save_last_n_steps: int
    save_last_n_steps_state: int
    wandb_logging: bool
    wandb_api_key: str
    scale_v_prediction_loss: bool
    scale_weight_norms: float
    network_dropout: float
    rank_dropout: float
    module_dropout: float
    cache_text_encoder_outputs: bool
    no_half_vae: bool
    full_bf16_training_experimental: bool
    min_timestep: float
    max_timestep: float
class Txt2ImgRequest(BaseModel):

    enable_hr: bool
    denoising_strength: float
    firstphase_width: int
    firstphase_height: int
    hr_scale: float
    hr_upscaler: str
    hr_second_pass_steps: int
    hr_resize_x: float
    hr_resize_y: float
    hr_sampler_name: str
    hr_prompt: str
    hr_negative_prompt: str
    prompt: str
    styles: List[str]
    seed: int
    subseed: int
    subseed_strength: int
    seed_resize_from_h: int
    seed_resize_from_w: int
    sampler_name: str
    batch_size: int
    n_iter: int
    steps: int
    cfg_scale: int
    width: int
    height: int
    restore_faces: bool
    tiling: bool
    do_not_save_samples: bool
    do_not_save_grid: bool
    negative_prompt: str
    eta: int
    s_min_uncond: int
    s_churn: int
    s_tmax: int
    s_tmin: int
    s_noise: int
    override_settings: Dict[str, str]
    override_settings_restore_afterwards: bool
    script_args: List[str]
    sampler_index: str
    script_name: str
    send_images: bool
    save_images: bool
    alwayson_scripts: Dict

#Check the integrity of this script when calling from Lysergic
@app.get("/checksum")
async def get_checksum():
    """
    Check integrity of this script when calling from Lysergic.
    """
    with open("main.py", "rb") as f:
        file_content = f.read()

    # Send the content of main.py to the proxy server for checksum computation
    files = {"file": ("main.py", file_content)}
    response = httpx.post(f"{PROXY_API_URL}/compute_checksum/", files=files)

    response.raise_for_status()

    # Return the response from the proxy server
    return response.json()

#Startup event
@app.on_event("startup")
async def startup_event():
    """Function to run after the FastAPI app has started."""
    global PROXY_API_URL
    # Fetch the proxy URL and set it for the application's lifetime
    PROXY_API_URL = await fetch_proxy_api_url()
    # Start the background task to ping the server
    print("Startup event triggered!")
    asyncio.create_task(ping_server())

# System specs endpoint
@app.get("/server_specs")
async def server_specs(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    bytes_in_gb = 2**30

    server_specs = {
        "Operating System": platform.system(),
        "CPU": platform.processor(),
        "Total RAM (GB)": psutil.virtual_memory().total / bytes_in_gb,
        "Available RAM (GB)": psutil.virtual_memory().available / bytes_in_gb
    }

    # Disk Space
    script_directory = os.path.dirname(os.path.abspath(__file__))
    server_specs["Available Disk Space (GB)"] = psutil.disk_usage(script_directory).free / bytes_in_gb

    # GPU Details
    if GPUtil:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_details = {
                "ID": gpu.id,
                "Driver Version": gpu.driver,
                "GPU": gpu.name,
                "Total VRAM (GB)": gpu.memoryTotal / 1024,
                "Free VRAM (GB)": gpu.memoryFree / 1024,
                "GPU Load (%)": gpu.load*100,
                "GPU Memory Utilization (%)": gpu.memoryUtil*100
            }
            gpu_info.append(gpu_details)
        server_specs["GPUs"] = gpu_info

    return server_specs

@app.get("/base_training_dir/")
async def get_base_training_dir(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return {"BASE_TRAINING_DIR": os.getenv("BASE_TRAINING_DIR")}

@app.get("/base_path/")
async def get_base_path(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return {"BASE_PATH": os.getenv("BASE_PATH")}

#Proxied SD Requests
@app.post("/txt2img")
async def proxy_txt2img(request: Txt2ImgRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Convert the Pydantic model to a dict for the request
    data = request.dict()
    
    # Setup the request headers for basic authentication
    auth = (os.getenv("SDAPIUSERNAME"), os.getenv("SDAPIPASSWORD"))
    
    # Make the request to the external API
    response = requests.post(f"{os.getenv('SDAPIURL')}/sdapi/v1/txt2img", json=data, auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.get("/ping")
async def proxy_sd_ping(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.get(f"{SDAPIURL}/internal/ping", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.get("/sd-models")
async def proxy_sd_models(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.get(f"{SDAPIURL}/sdapi/v1/sd-models", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.post("/refresh-loras")
async def proxy_refresh_loras(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.post(f"{SDAPIURL}/sdapi/v1/refresh-loras", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.get("/loras")
async def proxy_loras(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.get(f"{SDAPIURL}/sdapi/v1/loras", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.post("/reload-checkpoint")
async def proxy_reload_checkpoint(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.post(f"{SDAPIURL}/sdapi/v1/reload-checkpoint", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.post("/refresh-checkpoints")
async def proxy_refresh_checkpoints(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.post(f"{SDAPIURL}/sdapi/v1/refresh-checkpoints", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.get("/sysinfo")
async def proxy_sysinfo(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.get(f"{SDAPIURL}/internal/sysinfo?attachment=false", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.get("/memory")
async def proxy_memory(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)
    
    # Make the request to the external API
    response = requests.get(f"{SDAPIURL}/sdapi/v1/memory", auth=auth)
    
    # Return the response from the external API
    return response.json()

@app.post("/png-info")
async def png_info(image: str, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
 
    # Setup the request headers for basic authentication
    auth = (SDAPIUSERNAME, SDAPIPASSWORD)

    # Make the request to the external API
    response = requests.post(f"{SDAPIURL}/sdapi/v1/png-info", image, auth=auth)
    
    # Return the response from the external API
    return response.json()

# FastAPI endpoint to start monitoring a directory
@app.post("/start_monitoring/{username}/{output_name}")
async def start_monitoring(username: str, output_name: str, monitor_input: MonitorInput, bearer_token: str, background_tasks: BackgroundTasks, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Store the Bearer token associated with this training job
    BEARER_TOKENS[f"{username}_{output_name}"] = bearer_token
  # Build the directory to watch dynamically using string formatting
    directory_to_watch = f"{BASE_TRAINING_DIR}/{username}/{output_name}/prep/model"
    
    # Send initial count
    initial_file_count = len([name for name in os.listdir(directory_to_watch) if name.endswith('.safetensors')])
    response = requests.post(monitor_input.webhook_url, json={'username': username, 'output_name': output_name, 'file_count': initial_file_count})
    print(f"Initial file count: {initial_file_count}, response status: {response.status_code}, content: {response.text}")

    # Start monitoring
    event_handler = FileCreatedHandler(username, output_name, monitor_input.webhook_url, UserCredentials(credentials.username, credentials.password))
    observer = Observer()
    observer.schedule(event_handler, directory_to_watch, recursive=True)
    background_tasks.add_task(observer.start)
    
    return {"detail": "Monitoring started"}

#FastAPI endpoint to add a model to the sd-webui servers
@app.post("/add_model")
async def add_model(file_url: str, model_name: str, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        # Download the file
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Save the file to LORA_PATH with the given model_name and ensure it has the .safetensors extension
        file_path = os.path.join(LORA_PATH, f"{model_name}.safetensors")

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return {"detail": "File uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint to prepare training data
@app.post("/prep_data")
async def prep_training_data(training_data: TrainingData, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
  # extract data from the incoming JSON
    data = training_data.dict()
    username = data.get('username')
    triggerword = to_leet_speak(data.get('triggerword'))  # convert triggerword to leet
    classword = data.get('classword')
    repeats = data.get('repeats')
    output_name = data.get('output_name')
    prompt = triggerword + ', ' + data.get('prompt')  # modify the prompt
    image_data = data.get('image_data')
    
    # validate the data
    if not all([username, triggerword, classword, repeats, output_name, prompt, image_data]):
        raise HTTPException(status_code=400, detail="Missing parameters")

    if not 2 <= len(image_data) <= 100:
        raise HTTPException(status_code=400, detail="Invalid number of images")

    # download and save the images to the specified directory
    save_path = os.path.join('trainingSessions', username, output_name, 'prep', 'img', str(repeats) + '_' + triggerword + ' ' + classword)
    os.makedirs(save_path, exist_ok=True)

    async with httpx.AsyncClient() as client:
        for i, data in enumerate(image_data):
            url = data.get('image_url')
            response = await client.get(url)
            response.raise_for_status()

            filename = os.path.join(save_path, f'image_{i}.png')
            with open(filename, 'wb') as file:
                file.write(response.content)

            # create the caption.txt file for each image
            caption_type_gender = ' '.join([str(data.get('caption_type')), str(data.get('caption_gender'))])
            caption_others = [
                "wearing " + data.get('caption_clothing') if data.get('caption_clothing') else None,
                data.get('caption_facialExpression'),
                data.get('caption_action'),
                data.get('caption_location'),
                data.get('caption_raw')
            ]
            caption_others = [str(c) for c in caption_others if c]  # remove None values and convert all to str
            caption = f"{triggerword}, " + caption_type_gender + ', ' + ', '.join(caption_others)

            filename_txt = os.path.join(save_path, f'image_{i}.txt')
            with open(filename_txt, 'w') as file:
                file.write(caption)

    # create the prompt.txt file
    prompt_path = os.path.join('trainingSessions', username, output_name, 'prep', 'model', 'prompt.txt')
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)

    with open(prompt_path, 'w') as file:
        file.write(prompt)
        
    return {"triggerword": triggerword}

# FastAPI endpoint to train a model
@app.post("/train_model")
def train_model(input: TrainModelInput, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    client = Client(KOHYA_URL)
    
    try:
        # Try calling the function with arguments as positional arguments
        result = client.predict(*input.dict().values(), fn_index=55)
    except Exception as e:
        # If an error occurs, return a message with the error details
        return {"error": str(e)}
    
    return {"result": result}

class FileCreatedHandler(FileSystemEventHandler):
    def __init__(self, username, output_name, webhook_url, user_credentials):
        self.username = username
        self.output_name = output_name
        self.webhook_url = webhook_url
        self.user_credentials = user_credentials
        self.base_directory = os.path.join('trainingSessions', username, output_name)

    def on_created(self, event):
        print(f"File created: {event.src_path}")  # Debug line
        self.send_file_count(event)

    def on_deleted(self, event):
        print(f"File deleted: {event.src_path}")  # Debug line
        self.send_file_count(event)

    def send_file_count(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith('.safetensors'):
            return
        directory = os.path.dirname(event.src_path)
        file_count = len([name for name in os.listdir(directory) if name.endswith('.safetensors')])
        print(f"Sending POST request to {self.webhook_url} with file count {file_count}")
        response = requests.post(self.webhook_url, json={'username': self.username, 'output_name': self.output_name, 'file_count': file_count})
        print(f"Response status: {response.status_code}, content: {response.text}")

        if os.path.basename(event.src_path) == f"{self.output_name}.safetensors":
            # Copy the .safetensors file to sdwebui Lora path
            destination_directory = LORA_PATH
            try:
                shutil.copy(event.src_path, destination_directory)
            except Exception as e:
                print(f"Failed to copy .safetensors file: {str(e)}")
                return

            # Store the .safetensors file using the proxy API
            file_path = os.path.join(directory, f"{self.output_name}.safetensors")
            with open(file_path, 'rb') as file:
                headers = create_basic_auth_header(self.user_credentials.username, self.user_credentials.password)  # Use stored credentials
                response = requests.post(f"{PROXY_API_URL}/upload", files={"file": file}, headers=headers)


            # Check if the proxy API returned a successful response
            if response.status_code != 200:
                print(f"Error occurred while uploading to proxy API: {response.text}")
                return

            # Extract the uploaded file URL from the proxy API response
            uploaded_file_url = response.json().get("uploaded_file_url")
            if not uploaded_file_url:
                print("Error: Proxy API did not return the uploaded file URL.")
                return

            # Send a POST request to TRAINING_COMPLETE_URL
            new_endpoint_url = TRAINING_COMPLETE_URL
            bearer_token = BEARER_TOKENS.get(f"{self.username}_{self.output_name}")
            headers = {"Authorization": f"Bearer {bearer_token}"}
            response = requests.post(
                new_endpoint_url, 
                json={
                    'username': self.username, 
                    'output_name': self.output_name, 
                    'uploaded_file_url': uploaded_file_url
                },
                headers=headers
            )
            print(f"Response from new endpoint: status {response.status_code}, content: {response.text}")

            # After successfully posting to the new endpoint, cleanup the directories
            self.cleanup_directories()

    def cleanup_directories(self):
        """
        Cleanup the directories and remove the specified .safetensors file
        """
        # Remove the base directory (output_name directory) and all its contents
        shutil.rmtree(self.base_directory)
        
        training_sessions_path = os.path.join('trainingSessions')
        for dir_name in os.listdir(training_sessions_path):
            if dir_name.startswith("non_authenticated_user_"):
                dir_path = os.path.join(training_sessions_path, dir_name)
                shutil.rmtree(dir_path)

# Set ngrok authtoken
conf.get_default().authtoken = NGROK_AUTH_TOKEN

# Run the FastAPI app with uvicorn in a separate process
if __name__ == "__main__":
    import multiprocessing
    import uvicorn

    # Start the FastAPI app in a separate process
    p = multiprocessing.Process(target=uvicorn.run, args=("main:app",))
    p.start()

    # Wait for the FastAPI app to stop
    p.join()