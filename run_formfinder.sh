#!/bin/bash -x

# Define project directory and log file
PROJECT_DIR="/home/wilkins/FormFinder"
LOG_FILE="$PROJECT_DIR/data/logs/cron_runner.log"

# Ensure this script itself doesn't create runaway logs if it's misconfigured
# and called repeatedly by cron before PROJECT_DIR is accessible.
if [ ! -d "$PROJECT_DIR" ]; then
    # Fallback log location if project dir is not accessible initially
    # This might happen if cron runs before a mount point for /home/wilkins is ready,
    # though unlikely for a standard home directory.
    echo "$(date): CRITICAL - Project directory $PROJECT_DIR not found. Cannot proceed." >> /tmp/formfinder_bootstrap_error.log
    exit 1
fi

cd "$PROJECT_DIR" || { echo "$(date): CRITICAL - Failed to cd to $PROJECT_DIR. Exiting." >> "$LOG_FILE"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/data/logs"

# Function to append messages to log
log_message() {
    # Adding a timestamp to each message logged by this function
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "============================================================================"
log_message "Starting FormFinder process."

# Initialize and Activate Conda Environment
CONDA_ENV_NAME="formfinder"
CONDA_ACTIVATED_SUCCESSFULLY=false

# Try to find conda base path
CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE_PATH" ]; then
    # Common fallback paths for conda
    if [ -d "/home/wilkins/miniconda3" ]; then
        CONDA_BASE_PATH="/home/wilkins/miniconda3"
    elif [ -d "/home/wilkins/anaconda3" ]; then
        CONDA_BASE_PATH="/home/wilkins/anaconda3"
    fi
fi

if [ -n "$CONDA_BASE_PATH" ] && [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    log_message "Sourcing Conda from $CONDA_BASE_PATH/etc/profile.d/conda.sh"
    # shellcheck source=/dev/null
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
    
    log_message "Attempting to activate conda environment: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME"
    if [ $? -eq 0 ] && [ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]; then
        log_message "Conda environment '$CONDA_ENV_NAME' activated successfully."
        log_message "Python executable: $(which python3)"
        log_message "Active Conda environment: $CONDA_DEFAULT_ENV"
        CONDA_ACTIVATED_SUCCESSFULLY=true
    else
        log_message "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'."
        log_message "Current CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
        # Exit if Conda activation is critical
        # exit 1 
    fi
else
    log_message "WARNING: Conda initialization script not found or CONDA_BASE_PATH couldn't be determined. Python scripts might use system Python."
    log_message "Python executable: $(which python3)"
fi

# --- Script Execution ---
execute_python_script() {
    local script_name="$1"
    log_message "--- Running $script_name ---"
    # Use the python3 from the activated conda env (if active) or system path
    python3 "$script_name" >> "$LOG_FILE" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        log_message "ERROR: $script_name failed with status $status"
        # Deactivate Conda environment if it was activated
        if [ "$CONDA_ACTIVATED_SUCCESSFULLY" = true ] && [ ! -z "$CONDA_DEFAULT_ENV" ]; then
            log_message "Deactivating Conda environment due to script failure."
            conda deactivate
        fi
        exit $status # Exit script with the status of the failed python script
    fi
    log_message "$script_name completed successfully."
    return 0
}

execute_python_script "DataFetcher.py"
execute_python_script "DataProcessor.py"
execute_python_script "PredictorOutputter.py"
PREDICTOR_SUCCESS=$? # PredictorOutputter.py was the last critical script

if [ $PREDICTOR_SUCCESS -eq 0 ]; then
    execute_python_script "notifier.py"
else
    log_message "PredictorOutputter.py failed. Skipping notifications."
fi

log_message "FormFinder process completed."
log_message "============================================================================"

# Deactivate Conda environment if it was activated
if [ "$CONDA_ACTIVATED_SUCCESSFULLY" = true ] && [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    log_message "Deactivating Conda environment at the end of the script."
    conda deactivate
fi
