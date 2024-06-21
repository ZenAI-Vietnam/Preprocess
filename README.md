# Preprocess Images with Better Captions, Tag and IQA

## Install
```bash
python -m venv venv
source venv/bin/activate 

# Clone repo
git clone https://github.com/ZenAI-Vietnam/Preprocess.git
cd Preprocess
pip install -r requirements.txt
```
### Choose caption_mode
```bash
directories:
    ...
iqa:
    ...
    metric_name: null #if not use
    ...
caption:
    ...
    caption_mode: null #if not use 
    #choose model caption: 
    #   gpt == Lin-Chen/ShareGPT4V-7B;
    #   captioner == Lin-Chen/ShareCaptioner;
    #   tagger == trongg/swinv2_base;
questions:
    ...
    #fill your question
```
## Run
```bash
python preprocess.py --config-file /path/to/test.yaml
```