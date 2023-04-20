import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("pay_table.csv")

train_data, temp_data = train_test_split(df, test_size=0.15, random_state=1)

validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=1)

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "the_key_of_your_GC_service_account.json"

from google.cloud import aiplatform, storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {bucket_name}.")

project_id = "your_project_id_here"
bucket_name = "your_correspondent_GC_Storage_bucket_name_here"
source_file_name = "updatedPaytable.csv"
destination_blob_name = "updatedPaytable.csv"
location = "the_location_of_your_project_and_GC_Storage_bucket"

upload_blob(bucket_name, source_file_name, destination_blob_name)

aiplatform.init(project=project_id, location=location)
#
dataset = aiplatform.TabularDataset.create(
    display_name="updatedPaytable",
    gcs_source=[f"gs://{bucket_name}/{destination_blob_name}"],
    sync=True
)

if dataset.resource_name:
    print(f"Dataset created successfully with resource name: {dataset.resource_name}")
else:
    print("Dataset creation failed.")

dataset_id = dataset.resource_name
print("this is the id of the dataset:", dataset_id)

training_task_definition = "gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_tables_1.0.0.yaml"

training_task_inputs = {
    "targetColumn": "Overtimepay",
    "optimizationObjective": "MINIMIZE_MAE",
    "trainBudgetMilliNodeHours": 1000,
    "disableEarlyStopping": False,
}
from datetime import datetime

now = datetime.now()

training_pipeline = {
    "display_name": f"training_pipeline-{now.strftime('%Y%m%d%H%M%S')}",
    "input_data_config": {"dataset_id": dataset_id},
    "model_to_upload": {"display_name": f"model-{now.strftime('%Y%m%d%H%M%S')}"},
    "training_task_definition": training_task_definition,
    "training_task_inputs": training_task_inputs,
}

try:
    pipeline = aiplatform.PipelineJob(training_pipeline)
    pipeline.run()
except Exception as e:
    print(f"Error running training pipeline: {e}")

job_id = pipeline.job_id
job_client = aiplatform.JobServiceClient()
job = job_client.get_job(name=job_id)

print(job.state)
print(job.error_message)




