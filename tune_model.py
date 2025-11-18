import vertexai
from vertexai.preview.tuning import sft
from datetime import datetime

# 1. Setup Configuration
# Replace with your actual Project ID and Bucket Name
PROJECT_ID = "gcp-yard"
LOCATION = "us-central1" 
BUCKET_NAME = "medgemma-tuning-demo"

# The GCS path to the file you uploaded in the previous step
TRAINING_DATA_PATH = f"gs://{BUCKET_NAME}/training_data/data.jsonl"

# We append a timestamp so the job name is unique every time you run it
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
JOB_NAME = f"medgemma-shorthand-tuning-{TIMESTAMP}"

# 2. Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

print(f"Launching training job: {JOB_NAME}")

# 3. Launch the Supervised Fine-Tuning (SFT) Job
# Note: This uses the Gemma 2B model which is fast and cheap for demos.
sft_tuning_job = sft.train(
    source_model="google/gemma3@medgemma-27b-it",  # The base model we are tuning
    train_dataset=TRAINING_DATA_PATH,
    # Optional: Validation dataset can be added here if you had one
    epochs=4,          # How many times the model sees your data
    learning_rate_multiplier=3e-4, # How fast the model adapts (standard for LoRA)
    adapter_size=4,    # LoRA rank (smaller = fewer parameters to train)
    tuned_model_display_name=JOB_NAME
)

# 4. Monitor
print("Job submitted. You can close this script, or wait for it to finish.")
# While the job runs remotely on Google's GPUs, this line blocks your local script 
# until it's done. You can remove it if you want to 'fire and forget'.
sft_tuning_job.wait()

print("Tuning complete!")
print(f"Tuned model resource name: {sft_tuning_job.tuned_model_name}")
print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")