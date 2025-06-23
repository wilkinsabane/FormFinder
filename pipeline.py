import asyncio
from prefect import flow, get_run_logger

# Import tasks from their respective modules
# Assuming sdata_config.json, sdata_init_config.json, notifier_config.json are still in root for now
# These will be replaced by a central config.yaml later
from DataFetcher import run_data_fetcher_task
from DataProcessor import run_data_processor_task
from PredictorOutputter import run_predictor_outputter_task
from notifier import run_notifier_task

@flow(name="FormFinder ETL Pipeline", log_prints=True)
async def formfinder_pipeline(days_ahead_prediction: int = 7, predictions_output_dir: str = 'data/predictions'):
    logger = get_run_logger()
    logger.info("Starting FormFinder ETL Pipeline...")

    # 1. Fetch Data
    # Wait for data_fetch_result if you need to ensure it completes before subsequent tasks.
    # If run_data_fetcher_task is async, it will return a future-like object.
    # If it's synchronous wrapped by Prefect, Prefect handles the await.
    # Since run_data_fetcher_task is an async def, we should await it.
    logger.info("Kicking off Data Fetcher task...")
    data_fetch_result = await run_data_fetcher_task.submit()
    # .submit() makes it run in the background if using a task runner that supports concurrency.
    # We need to wait for its completion if subsequent tasks depend on it.
    await data_fetch_result.wait() # Explicitly wait for the async task to complete
    logger.info("Data Fetcher task finished.")

    # 2. Process Data
    # This task is synchronous, Prefect handles its execution.
    # It depends on the successful completion of data fetching.
    logger.info("Kicking off Data Processor task...")
    # Pass data_fetch_result as an upstream dependency if needed, or rely on sequential execution.
    # For now, Prefect's default sequential execution for tasks called with .submit() and then .wait() or direct call is fine.
    # Or, to make dependency explicit:
    data_process_result = run_data_processor_task.submit(wait_for=[data_fetch_result])
    data_process_result.wait()
    logger.info("Data Processor task finished.")

    # 3. Generate Predictions
    # Depends on data processing.
    logger.info("Kicking off Predictor Outputter task...")
    predictor_output_result = run_predictor_outputter_task.submit(
        days_ahead_pred=days_ahead_prediction,
        pred_output_dir=predictions_output_dir,
        wait_for=[data_process_result]
    )
    predictor_output_result.wait()
    logger.info("Predictor Outputter task finished.")

    # 4. Notify
    # Depends on predictions being generated.
    logger.info("Kicking off Notifier task...")
    notifier_result = run_notifier_task.submit(wait_for=[predictor_output_result])
    notifier_result.wait()
    logger.info("Notifier task finished.")

    logger.info("FormFinder ETL Pipeline completed successfully!")

async def main_async(): # Wrapper for asyncio.run
    # Parameters can be passed here or loaded from a config file for the flow
    await formfinder_pipeline(days_ahead_prediction=7, predictions_output_dir='data/predictions')

if __name__ == "__main__":
    # To run the pipeline:
    # prefect deployment build ./pipeline.py:formfinder_pipeline -n formfinder-deployment -q default --path /home/wilkins/FormFinder --apply
    # prefect agent start -q default
    # Or, for a local test run:
    asyncio.run(main_async())
    print("Pipeline execution script finished. Check Prefect UI or logs for flow run status.")
