This directory contains code to run neurologic decoding on top of BART and T5. 
You need to choose one to decode and comment out the other in [beam_search.sh](beam_search.sh) and [decode.sh](decode.sh)

* Activate conda environment
    ```
    conda activate hug
    export PYTHONPATH=${HOME_PATH}/neurologic_decoding
    ```

* Run beam search baseline 
    ```
    mkdir OUTPUT_DIR
    bash beam_search.sh DEVICE_ID SPLIT MODEL_PATH OUTPUT_DIR OUTPUT_FILE_NAME
    ```

* Run neurologic decoding
    ```
    bash decode.sh DEVICE_ID SPLIT MODEL_PATH OUTPUT_DIR OUTPUT_FILE_NAME
    ```
  
* Format the output file for evaluation via [expand.py](../expand.py)
    ```
    cd ..
    python expand.py --split SPLIT --input_file OUTPUT_FILE_NAME --output_file FORMAT_FILE_NAME
    ```