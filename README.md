## Getting Start

**[Optional] Get access to backdoor models and base model fine-tuning dataset from Huggingface**

If you want to use the backdoor models and base model fine-tuning dataset, please ensure you have permission to them. To authenticate with Hugging Face, use one of the following methods:

**Option 1: Interactive Login**
```bash
huggingface-cli login
```
Then enter your Hugging Face token (starting with "hf_") when prompted.

**Option 2: Environment Variable**
```bash
export HF_TOKEN="your_huggingface_token"
```
Replace `your_huggingface_token` with your actual Hugging Face token (starting with "hf_").

## Get Code
```bash
python defense.py --attack [YOUR_ATTACKER_NAME] --defense [YOUR_DEFENDER_NAME] 
python calculate_ASR.py --attack [YOUR_ATTACKER_NAME] --defense [YOUR_DEFENDER_NAME]

```

## ANSWER FILES ARE : 

args.answer_file = f"result/{args.attack}_{args.defense}.jsonl"

Supports:
- **Attacker**:
  - VPI-SS
  - VPI-CI
  - AutoPoison
  - CB-MT
  - CB-ST

- **Defender**:
  - driftwatch
  - no_defense

Don't forget to **add your openai api** to calculate ASR for CB-MT, CB-ST, and VPI-SS.

## For Visualization and plotting 

```bash

export HF_TOKEN= ""
python visual.py

```