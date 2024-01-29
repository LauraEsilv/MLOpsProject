# To complete
import time
from prefect import flow, serve

if __name__ == "__main__":
    train_model_workflow = train_model_workflow.to_deployment(name="train_model")
    batch_predict_workflow = batch_predict_workflow.to_deployment(name="batch_predict")
    serve(train_model_workflow, batch_predict_workflow)