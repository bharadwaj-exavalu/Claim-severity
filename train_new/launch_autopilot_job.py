import boto3
import time

def launch_autopilot_job(job_name, bucket_name, dataset_s3_path, target_column, output_bucket, role_arn):
    """Launches an AWS SageMaker Autopilot job with cost-effective instance settings."""

    sagemaker_client = boto3.client("sagemaker")

    try:
        response = sagemaker_client.create_auto_ml_job(
            AutoMLJobName=job_name,
            InputDataConfig=[
                {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": dataset_s3_path,
                        }
                    },
                    "TargetAttributeName": target_column
                }
            ],
            OutputDataConfig={"S3OutputPath": f"s3://{output_bucket}/"},
            ProblemType="Regression",  # Adjust if needed
            AutoMLJobObjective={"MetricName": "MAE"},
            RoleArn=role_arn,
            AutoMLJobConfig={
                "CompletionCriteria": {
                    "MaxRuntimePerTrainingJobInSeconds": 3600,  # 1 hour max runtime per job
                    "MaxAutoMLJobRuntimeInSeconds": 10800  # 3 hours total job runtime
                },
                "SecurityConfig": {
                    "EnableInterContainerTrafficEncryption": False,
                    "VolumeKmsKeyId": "",  # Add KMS key if encryption is required
                },
                "CandidateGenerationConfig": {
                    "FeatureSpecificationS3Uri": "",  # Optional, for feature engineering customization
                },
                "TrainingJobConfig": {
                    "ResourceConfig": {
                        "InstanceType": "ml.t3.medium",  # Cost-effective instance type
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 10  # Reduce storage to control cost
                    }
                }
            }
        )

        print(f"✅ Autopilot job '{job_name}' started successfully!")
        return response["AutoMLJobArn"]

    except Exception as e:
        print(f"❌ Error launching Autopilot job: {e}")
        return None

def monitor_autopilot_job(job_name):
    """Monitors the progress of an Autopilot job."""
    sagemaker_client = boto3.client("sagemaker")

    print(f"Monitoring Autopilot job '{job_name}'...")
    status = "InProgress"

    while status == "InProgress":
        response = sagemaker_client.describe_auto_ml_job(AutoMLJobName=job_name)
        status = response["AutoMLJobStatus"]

        print(f"Current status: {status}")
        if status == "InProgress":
            time.sleep(60)  # Check every minute

    if status == "Completed":
        print(f"✅ Autopilot job completed successfully!")
        best_candidate = response.get("BestCandidate", {})
        if best_candidate:
            print(f"Best model container image: {best_candidate.get('InferenceContainers', [{}])[0].get('Image', 'N/A')}")
            print(f"Best model objective metric: {best_candidate.get('FinalAutoMLJobObjectiveMetric', {}).get('Value', 'N/A')}")
    else:
        print(f"⚠️ Autopilot job ended with status: {status}")

    return status

if __name__ == "__main__":
    # Configuration
    job_name = f"autopilot-job-{int(time.time())}"  # Unique job name
    bucket_name = "claim-severity-trainnew"
    dataset_s3_path = f"s3://{bucket_name}/synthetic_claim_data.csv"
    target_column = "Claim_Amount"
    output_bucket = bucket_name
    role_arn = "arn:aws:iam::686583312671:role/SageMaker-AutopilotRole-New"  # Using the provided role

    # Launch the Autopilot job
    job_arn = launch_autopilot_job(job_name, bucket_name, dataset_s3_path, target_column, output_bucket, role_arn)

    if job_arn:
        print("Do you want to monitor the job progress? (y/n)")
        response = input().lower()
        if response == 'y':
            monitor_autopilot_job(job_name)
        else:
            print(f"You can monitor the job later using: python monitor_autopilot_job.py {job_name}")
    else:
        print("Failed to launch Autopilot job.")
