# import dagshub
# import mlflow
# from dotenv import load_dotenv

# load_dotenv(override=True)


# # This one line handles the URI and authentication for you
# dagshub.init(repo_owner="krishnabhagavan910", repo_name="generic-mlops-pipeline")

# # Now MLflow is "connected" to the remote server
# print(f"Tracking URI: {mlflow.get_tracking_uri()}")


# from mlflow.tracking import MlflowClient

# client = MlflowClient()

# # List of model names to delete
# models_to_delete = ["construction_duration", "construction_risk", "my_awesome_model"]

# for model_name in models_to_delete:
#     try:
#         client.delete_registered_model(name=model_name)
#         print(f"Successfully deleted: {model_name}")
#     except Exception as e:
#         print(f"Error deleting {model_name}: {e}")