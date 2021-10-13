This directory contains code to run neurologic decoding on top of GPT2

* Activate conda environment
    ```
    conda activate hug
    export PYTHONPATH=${HOME_PATH}/neurologic_decoding
    ```

* Run beam search baseline 
    ```
    bash beam_search.sh DEVICE_ID SPLIT MODEL_PATH OUTPUT_FILE_NAME
    ```

* Run neurologic decoding
    ```
    bash decode.sh DEVICE_ID SPLIT MODEL_PATH OUTPUT_FILE_NAME
    ```
  
* Format the output file for evaluation via [expand.py](../expand.py)
    ```
    cd ..
    python expand.py --split SPLIT --input_file OUTPUT_FILE_NAME --output_file FORMAT_FILE_NAME
    ```