# LDEDE

### LRP-Driven Efficient Detection and Editing Framework for LLM Privacy Neurons
A framework for locating privacy-related neurons in large language models with **Layer-wise Relevance Propagation (LRP)** and evaluating targeted neuron suppression.

The repository provides experiments on **personal names, phone numbers, and email text**, together with Integrated Gradients baselines and analyses of interactions between privacy categories.

## Overview

LDEDE studies which feed-forward network components are associated with memorized private information and how suppressing selected components changes privacy and language-modeling performance.

The experimental workflow consists of four stages:

1. **Identify memorized samples.** Score candidate samples using name MRR, phone Exposure, or email-text perplexity.
2. **Compute relevance scores.** Apply LRP and retain signed scores for each sample and layer.
3. **Aggregate across samples.** Select positive-relevance candidates, count their selection frequency, and construct a layer-to-neuron index mapping.
4. **Suppress and evaluate.** Apply forward hooks to selected FFN activations and compare privacy metrics and validation perplexity before and after suppression.

```text
Corpus → Preprocessing → Model fine-tuning → Memorized samples
                                                  │
                                                  ▼
                                      Per-sample LRP scores
                                                  │
                                                  ▼
                                      Frequency-based selection
                                                  │
                                                  ▼
                                      Targeted FFN suppression
                                                  │
                                                  ▼
                                  Privacy metrics + Validation PPL
```

## Repository structure

```text
LDEDE/
├── data_process/
│   ├── enron.txt                           # Dataset source references
│   ├── enron_preprocess.py                 # Email preprocessing and text splits
│   ├── txt_fast_to_arrow.py                # Tokenization and Arrow datasets
│   ├── check_arrow.py
│   └── decode_check.py
├── fine_tuning/
│   ├── run_clm_no_trainer.py               # Causal language model fine-tuning
│   ├── accelerate_cli.py                   # Configured Accelerate launcher
│   └── run_train.sh                        # Linux training wrapper
├── main/
│   ├── name/                              # Personal-name experiments
│   ├── phone/                             # Phone-number experiments
│   ├── email/                             # Email-text experiments
│   │   # Each category contains sample mining, LRP, aggregation,
│   │   # editing, and a base_line/ directory.
│   └── Seesaw/
│       ├── neurons_process/               # Cross-category neuron analyses
│       └── sample_seesaw/                  # Sample grouping and editing analyses
├── requirements.txt
└── README.md
```

The following sections describe how to prepare model checkpoints, corpora, Arrow datasets, and neuron files.

## Environment setup

The research environment uses **Python 3.10**. Use an NVIDIA GPU for fine-tuning and attribution. The main evaluation scripts select CUDA when available and support CPU execution.

```bash
git clone https://github.com/YS-SM/LDEDE.git
cd LDEDE
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

The checked-in `requirements.txt` records the research environment. Its principal entries include:

| Package | Recorded version |
| --- | --- |
| PyTorch | `2.1.1+cu118` |
| Transformers | `4.57.1` |
| Captum | `0.5.0` |
| Accelerate | `0.14.0` |
| Datasets | `4.3.0` |
| NumPy | `1.26.4` |
| NLTK | `3.8.1` |

Copy `requirements.txt` to `requirements.local.txt` and configure dependencies for your platform. Match the Torch build to your CUDA environment, align the Torch/Transformers and Datasets/Dill/Multiprocess versions, and configure installation sources for `baukit` and `triton`. Then install:

```bash
python -m pip install --upgrade pip
# After preparing requirements.local.txt:
python -m pip install -r requirements.local.txt
python -m pip check
python -c "import torch; from transformers import GPT2LMHeadModel, GPT2Tokenizer; from captum.attr import LRP; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

The name-extraction script uses NLTK tokenization, tagging, and named-entity resources and downloads them as needed. For offline execution, download these resources in advance.

## Data and model preparation

### 1. Prepare the corpus

Dataset references are recorded in `data_process/enron.txt`. Organize the Enron corpus as a **maildir-style directory of individual email files** for preprocessing.

Set the following constants in `data_process/enron_preprocess.py`:

| Setting | Description |
| --- | --- |
| `MAIL_DIR` | Input email directory |
| `OUTPUT_DATA_DIR` | Directory for `train.txt`, `valid.txt`, and `test.txt` |
| `OUTPUT_TEL_PATH` | Output file for extracted telephone information |
| `MAX_WORKERS` | Positive worker count suitable for your CPU |

```bash
python data_process/enron_preprocess.py
```

### 2. Prepare tokenized datasets

In `data_process/txt_fast_to_arrow.py`, set `TOKENIZER_PATH`, `TXT_FILES`, and `OUTPUT_ROOT`. Include **both the training and validation text files** in `TXT_FILES` and set `BLOCK_SIZE = 512` to match the main editing scripts.

```bash
python data_process/txt_fast_to_arrow.py
```

For inputs named `train.txt` and `valid.txt`, the converter writes:

```text
<OUTPUT_ROOT>/
├── train/final/
└── valid/final/
```

Each dataset contains `input_ids`, `attention_mask`, and `labels`. Padding positions in the labels are set to `-100`.

### 3. Fine-tune or provide a checkpoint

Prepare a GPT-2 checkpoint and its tokenizer. Use `fine_tuning/run_clm_no_trainer.py` for fine-tuning. Example single-process configuration:

```bash
python fine_tuning/run_clm_no_trainer.py --model_name_or_path "<BASE_GPT2>" --train_file "<ARROW_ROOT>/train/final" --validation_file "<ARROW_ROOT>/valid/final" --output_dir "<RUN_OUTPUT>" --block_size 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 5e-5 --num_train_epochs 1 --seed 42 --keep_final_model
```

Replace every `<...>` placeholder before running. With `--keep_final_model`, the final model and tokenizer are saved under `<RUN_OUTPUT>/final_model`. Use the same selected checkpoint throughout sample mining, localization, and evaluation.

For distributed training, adapt `fine_tuning/accelerate_cli.py` and `fine_tuning/run_train.sh`. Their supplied configuration contains Linux-specific absolute paths, four processes, BF16 precision, and a GPT-2 XL model path.

## Run the main experiments

Run commands from the repository root. The sample-mining scripts accept CLI arguments; localization, aggregation, and editing generally use configuration dictionaries inside the Python files. Replace the supplied `D:\expt\...` paths before execution.

Configure sample labels, neuron keys, and intermediate filenames using [Data formats](#data-formats).

### 1. Mine memorized samples

```bash
# Phone numbers: filter by Exposure
python main/phone/memorized_phone.py --model_name_or_path "<CHECKPOINT>" --raw_text_path "<TEXT_ROOT>/train.txt" --output_path "main/phone/gpt2_memorized_phone.txt" --threshold 14.0

# Personal names: filter by mean reciprocal rank
python main/name/memorized_NAME.py --model_name_or_path "<CHECKPOINT>" --raw_text_path "<TEXT_ROOT>/train.txt" --output_path "main/name/gpt2_memorized_privacy.txt" --threshold 0.4

# Email text: filter by target-text perplexity
python main/email/memorized_email.py --model_name_or_path "<CHECKPOINT>" --raw_text_path "<TEXT_ROOT>/train.txt" --output_path "main/email/gpt2_high_memory_emails_generation.txt" --ppl_threshold 10.0
```

Adjust `--threshold` and `--ppl_threshold` to select the memorization level for your experiment. Run each script with `--help` to view all mining options.

### 2. Compute LRP scores

In the relevant `lrp_privacy_neuron_localization_npy.py`, configure the sample path, `gpt2_path`, `save_npy_path`, sample limit, threshold, and batch size. Then run the categories you need:

```bash
python main/phone/lrp_privacy_neuron_localization_npy.py
python main/name/lrp_privacy_neuron_localization_npy.py
python main/email/lrp_privacy_neuron_localization_npy.py
```

The output is a NumPy object array of sample records containing metadata and `privacy_neuron_scores`, a mapping from layer indices to signed score arrays.

The localization scripts use GPT-2 with 12 layers and 3,072 FFN channels per layer. Attribution operates on FFN inputs: the 768-dimensional score vector is tiled four times to obtain 3,072 entries. To use another architecture, update the module references and dimensions accordingly.

### 3. Aggregate neuron candidates

Set `lrp_npy_path` to the exact output of the preceding stage and choose `save_neuron_path` in each `lrp_privacy_neuron_localization_FQ.py`.

```bash
python main/phone/lrp_privacy_neuron_localization_FQ.py
python main/name/lrp_privacy_neuron_localization_FQ.py
python main/email/lrp_privacy_neuron_localization_FQ.py
```

Phone aggregation parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `top_z` | `5000` | Maximum number of selected neurons |
| `global_top_ratio` | `0.5` | Fraction of positive-score candidates selected per sample |
| `vote_ratio_threshold` | `0.45` | Minimum cross-sample selection frequency |
| `lrp_positive_threshold` | `1e-6` | Minimum positive relevance score |
| `layer_range` | `(0, 11)` | Included GPT-2 layers |
| `num_processes` | `20` | Aggregation worker count; adjust to available CPU resources |

Qualified neurons are ranked by selection frequency and mean relevance, with additional tie-breaking. If fewer than `top_z` neurons qualify, the script retains the available candidates. It writes `.npy` and `.json` files containing `neuron_list` and `compatibility_dict`.

### 4. Suppress neurons and evaluate

In each `edit.py`, set the model, sample, neuron, and validation dataset paths. Apply the neuron-key alignment below before loading newly aggregated files.

```bash
python main/phone/edit.py
python main/name/edit.py
python main/email/edit.py
```

Editing uses runtime forward hooks to suppress selected outputs of `model.transformer.h[layer].mlp.c_fc`. Set `noise_strength` to zero for activation zeroing, or enable `use_noise` with a positive strength to replace selected activations with noise. The intervention remains active during evaluation, and hooks are removed afterward.

| Evaluation | Desired direction after suppression | Interpretation |
| --- | --- | --- |
| Name MRR | Lower | Lower rank-based recoverability of target names |
| Phone Exposure | Lower | Lower rank-based phone memorization |
| Email target-text PPL | Higher | Lower likelihood of the memorized target text |
| Validation PPL | Small change | Preservation of language-modeling performance on the chosen validation set |

Report privacy metrics alongside validation PPL to evaluate the privacy–utility trade-off. Phone Exposure uses the product of per-digit ranks to estimate sequence rank.

## Data formats

Use the following mappings to connect sample mining, localization, aggregation, and editing:

| Interface | Writer and reader formats | Configuration |
| --- | --- | --- |
| Phone samples | Mining writes English labels such as `Original text:` and `Phone number:`; localization and editing expect Chinese labels and full-width colons. | Make both downstream parsers accept the writer's format, or convert the generated sample file to the expected format. |
| Email samples | Mining writes `[PPL: ...]`, `[Prompt]:`, `[Target text]:`, and `[Generation preview]:`; downstream code expects `【PPL：...】`, `【prompt】：`, `【target_text】：`, and `【generation_preview】：`. | Align labels, brackets, and punctuation in both readers with the generated file. |
| Neuron files | Aggregation writes `compatibility_dict`; the three main editing scripts read `兼容字典`. | Update the editing loaders to accept the emitted key, as illustrated below. |
| Phone score filename | Localization defaults to `gpt2_phone_lrp_proj_results_shengao.npy`; aggregation defaults to `gpt2_phone_lrp_proj_results.npy`. | Set `lrp_npy_path` equal to the actual `save_npy_path`. |

For the neuron file, replace the existing key check and lookup in `load_gpt2_privacy_neurons` with equivalent handling of both formats, retaining the remaining validation:

```python
npy_data = np.load(neuron_path, allow_pickle=True).item()
if not isinstance(npy_data, dict):
    raise ValueError("Expected a dictionary of neuron data")
privacy_neurons = npy_data.get("compatibility_dict")
if privacy_neurons is None:
    privacy_neurons = npy_data.get("兼容字典")
if not isinstance(privacy_neurons, dict) or not privacy_neurons:
    raise ValueError("Expected a non-empty layer-to-neuron mapping")
```

Check the parsed sample count and pass each stage's output path to the next stage.

## Baselines and Seesaw analyses

Each privacy category has a `base_line/` directory:

- `*_GA.py`: Integrated Gradients attribution.
- `*_JH.py`: aggregation of attribution results.
- `*_edit.py`: suppression and evaluation of the baseline-selected neurons. The email baseline editor is named `eamil_edit.py` in the repository.

Configure their paths and file formats separately before running them.

`main/Seesaw/` contains additional analyses of how neuron relevance and editing behavior vary across privacy categories. The `neurons_process/` scripts operate on relevance or neuron files, and `sample_seesaw/` contains sample grouping and editing analyses. These scripts have their own input paths and are intended to be run after preparing the corresponding main-experiment artifacts.

## Paper and citation

**LRP-Driven Efficient Detection and Editing Framework for LLM Privacy Neurons**

## Questions

Use this repository's **Issues** tab for questions and reproducibility reports. Include the script name, relevant settings, package versions, and traceback so that the run can be diagnosed.
