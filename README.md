# LDEDE

**LRP-Driven Efficient Detection and Editing Framework for LLM Privacy Neurons**

LDEDE uses **Layer-wise Relevance Propagation (LRP)** to locate privacy-related neurons and suppress their activations. The code includes experiments on **personal names, phone numbers, and email text**, Integrated Gradients baselines, and Seesaw analyses across privacy categories.

```text
Memorized samples → LRP attribution → Neuron selection → Suppression and evaluation
```

## Repository structure

```text
LDEDE/
├── data_process/       # Corpus preprocessing and tokenization
├── fine_tuning/        # Language model fine-tuning
├── main/
│   ├── name/           # Personal-name experiments
│   ├── phone/          # Phone-number experiments
│   ├── email/          # Email-text experiments
│   └── Seesaw/         # Cross-category analyses
└── requirements.txt
```

## Installation

Use **Python 3.10** and a CUDA-capable GPU.

```bash
git clone https://github.com/YS-SM/LDEDE.git
cd LDEDE
python -m venv .venv
```

Activate `.venv` with `source .venv/bin/activate` on Linux or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell.

Copy `requirements.txt` to `requirements.local.txt`. Configure the Torch/CUDA build and installation sources for `baukit` and `triton`, and align the Torch/Transformers and Datasets/Dill/Multiprocess versions. Then install:

```bash
python -m pip install -r requirements.local.txt
python -m pip check
```

## Datasets

Please credit the original dataset authors when using their data.

| Resource | Source | Format / purpose |
| --- | --- | --- |
| Enron Email Dataset | [CMU dataset page](https://www.cs.cmu.edu/~enron/) | Original email corpus organized in mail folders |
| Enron-Spam | [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) | CSV distribution referenced in this project's data notes |
| MIMIC guide | [NUIST-NIM/MIMIC-NLP-Intro](https://github.com/NUIST-NIM/MIMIC-NLP-Intro) | Additional dataset setup reference recorded in `data_process/enron.txt` |

The Enron preprocessing script reads individual email files. For the CSV distribution, export each email to a text file before preprocessing.

## Usage

Run commands from the repository root. Set the model, data, and output paths in each script's configuration before running.

### 1. Prepare data

In `enron_preprocess.py`, configure `MAIL_DIR`, `OUTPUT_DATA_DIR`, and `OUTPUT_TEL_PATH`. In `txt_fast_to_arrow.py`, configure `TOKENIZER_PATH`, `OUTPUT_ROOT`, and `TXT_FILES` with both `train.txt` and `valid.txt`. Set `BLOCK_SIZE = 512`.

```bash
python data_process/enron_preprocess.py
python data_process/txt_fast_to_arrow.py
```

### 2. Fine-tune the model

Prepare a GPT-2 checkpoint and tokenizer. Example training command; replace `<...>` with your paths:

```bash
python fine_tuning/run_clm_no_trainer.py --model_name_or_path "<BASE_GPT2>" --train_file "<ARROW_ROOT>/train/final" --validation_file "<ARROW_ROOT>/valid/final" --output_dir "<RUN_OUTPUT>" --block_size 512 --keep_final_model
```

Use `<RUN_OUTPUT>/final_model` for the following experiments. Distributed training configurations are provided in `fine_tuning/accelerate_cli.py` and `fine_tuning/run_train.sh`.

### 3. Locate and suppress privacy neurons

Example for phone numbers, after configuring paths and matching the sample labels and neuron-file keys between stages:

```bash
python main/phone/memorized_phone.py
python main/phone/lrp_privacy_neuron_localization_npy.py
python main/phone/lrp_privacy_neuron_localization_FQ.py
python main/phone/edit.py
```

Use the corresponding scripts in `main/name/` and `main/email/` for the other categories. Their sample-mining entry points are `memorized_NAME.py` and `memorized_email.py`.

Configure these connections between stages:

- Match phone and email sample labels and punctuation in the writers and parsers.
- Set aggregation's `lrp_npy_path` to localization's `save_npy_path`.
- Read aggregation's `compatibility_dict` key in the editing loaders, which use `兼容字典` in the supplied code.

The main localization scripts use GPT-2 with 12 layers. Editing suppresses selected FFN activations through runtime forward hooks. Adjust `top_z`, `global_top_ratio`, and `vote_ratio_threshold` in the aggregation scripts to control neuron selection.

## Evaluation

| Metric | Desired change after suppression |
| --- | --- |
| Name MRR | Decrease |
| Phone Exposure | Decrease |
| Email target-text perplexity | Increase |
| Validation perplexity | Remain close to the original model |

Integrated Gradients baselines are in each category's `base_line/` directory. Seesaw analyses are in `main/Seesaw/`.

## Acknowledgments

We thank the Enron dataset creators, the maintainers of the linked data resources, and the open-source communities behind [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), and [Captum](https://github.com/pytorch/captum).
