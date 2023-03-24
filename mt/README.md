This directory contains code to run neurologic decoding on machine translation

* Activate conda environment
    ```
    conda activate hug
    export PYTHONPATH=${HOME_PATH}/neurologic_decoding
    ```

* Run beam search baseline on machine translation
    ```
    bash beam_search.sh DEVICE_ID OUTPUT_FILE_NAME
    ```

* Run neurologic decoding on machine translation
    ```
    bash decode.sh DEVICE_ID OUTPUT_FILE_NAME
    ```
  
* For evaluation, please refer to the [repo from Stanovsky et al](https://github.com/gabrielStanovsky/mt_gender).