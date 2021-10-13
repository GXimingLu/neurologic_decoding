# Neurologic Decoding

This is the official repo for the paper ["Neurologic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints"](https://aclanthology.org/2021.naacl-main.339.pdf) (NAACL 2021)

## Requirement
We suggest using conda to setup environment. With conda installed, create three new environments:
1. Huggingface 
    ```
    conda create -n hug python=3.7
    conda activate hug
    pip install -r huggingface.txt
    ```
2. Fairseq
   
    You need to first replace ``HOME_PATH`` in [fairseq.yml](fairseq.yml) with your home path. 
    ```
    conda env create -f fairseq.yml
    ```
   You could also refer to installation instructions in [fairseq](https://github.com/pytorch/fairseq/tree/v0.10.1) repo.
2. Unilm

    You need to first replace ``HOME_PATH`` in [unilm.yml](unilm.yml) with your home path. 
    ```
    conda env create -f unilm.yml
    ```
   Then, you need to install a specific version of huggingface package which incorporates unilm following instructions [here](https://github.com/huggingface/transformers/pull/2160).
  
Our code is tested on Quadro RTX 8000 with CUDA version 11.2 and driver version 460.27.04

## Model
We release our models trained on CommonGen data for reproducibility. 
1. GPT2:  [https://drive.google.com/drive/folders/1Jqav26p_g6BmpNg-6mx0AMiPYHH07Vju?usp=sharing](https://drive.google.com/drive/folders/1Jqav26p_g6BmpNg-6mx0AMiPYHH07Vju?usp=sharing)
2. BART:  [https://drive.google.com/drive/folders/19UUug_dkZbSltw1P-CrH_G9gKQwBuAA7?usp=sharing](https://drive.google.com/drive/folders/19UUug_dkZbSltw1P-CrH_G9gKQwBuAA7?usp=sharing)
3. T5:    [https://drive.google.com/drive/folders/1sMQYbhHjp5p0MuzerQT8zS6yAXyIej5g?usp=sharing](https://drive.google.com/drive/folders/1sMQYbhHjp5p0MuzerQT8zS6yAXyIej5g?usp=sharing)
4. Unilm: [https://drive.google.com/drive/folders/1V6ynPZE5XgAOIkPtb7_Npskz7S0OLp1N?usp=sharing](https://drive.google.com/drive/folders/1V6ynPZE5XgAOIkPtb7_Npskz7S0OLp1N?usp=sharing)
5. Unilm_v2: [https://drive.google.com/drive/folders/1ukVAg2RaCQqZXh1RtkPGBCleW8Xah7u8?usp=sharing](https://drive.google.com/drive/folders/1ukVAg2RaCQqZXh1RtkPGBCleW8Xah7u8?usp=sharing)

Please follow instructions in each sub-directory to run neurologic decoding on top of each model. 

## Evaluation
We adapt evaluation script from [CommonGen](https://github.com/INK-USC/CommonGen/tree/master/evaluation) for automatic metrics.
If you couldn't install ROUGE successfully, please refer [here](https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu) for troubleshooting.

## Citation
If you use this codebase in your work, please consider citing our paper:
```
@inproceedings{lu-etal-2021-neurologic,
    title = "{N}euro{L}ogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints",
    author = "Lu, Ximing  and  West, Peter  and  Zellers, Rowan  and  Le Bras, Ronan  and  Bhagavatula, Chandra  and  Choi, Yejin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,    
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.339",
    pages = "4288--4299",
}
```