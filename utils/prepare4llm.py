from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import transformers


def get_desc(domain, lookback_len, pred_len):
    
    description = {
        "Agriculture": ["Retail Broiler Composite", "month"],
        "Climate": ["Drought Level", "week"],
        "Economy": ["International Trade Balance", "month"],
        "Energy": ["Gasoline Prices", "week"],
        "Environment": ["Air Quality Index", "day"],
        "Health_US": ["Influenza Patients Proportion", "week"],
        "Security": ["Disaster and Emergency Grants", "month"],
        "SocialGood": ["Unemployment Rate", "month"],
        "Traffic": ["Travel Volume", "month"],
        "copper": ["Copper Price", "month"]
    }

    [OT, freq] = description[domain]

    desc = (f"Below is historical reporting information over the past {lookback_len} {freq}s concerning the {OT}. " 
        f"Based on these reports, predict the potential trends and anomalies of the {OT} for the next {pred_len} {freq}s.")
    return desc

def get_llm(llm_model:str, llm_layers:int=0):
    if llm_model == 'llama':
        # llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
        llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        if llm_layers:
            llama_config.num_hidden_layers = llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        try:
            llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                local_files_only=True,
                config=llama_config,
                # load_in_4bit=True
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                local_files_only=False,
                config=llama_config,
                # load_in_4bit=True
            )
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'huggyllama/llama-7b',
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'huggyllama/llama-7b',
                local_files_only=False
            )
    elif llm_model == 'gpt2':
        gpt2_config = GPT2Config.from_pretrained('../llms/gpt2')
        if llm_layers:
            gpt2_config.num_hidden_layers = llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        try:
            llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                local_files_only=True,
                config=gpt2_config,
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                local_files_only=False,
                config=gpt2_config,
            )

        try:
            tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                local_files_only=False
            )
    elif llm_model == 'bert':
        hf_id = 'google-bert/bert-base-uncased'   # alias
        bert_config = BertConfig.from_pretrained(hf_id)
        if llm_layers:
            bert_config.num_hidden_layers = llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True

        # 先尝试本地，没有就联网下载（兜住所有异常）
        try:
            llm_model = BertModel.from_pretrained(
                hf_id,
                local_files_only=True,
                config=bert_config,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"[LLM] local load failed: {type(e).__name__}: {e} -> downloading…")
            llm_model = BertModel.from_pretrained(
                hf_id,
                local_files_only=False,
                config=bert_config,
                ignore_mismatched_sizes=True
            )

        try:
            tokenizer = BertTokenizer.from_pretrained(hf_id, local_files_only=True)
        except Exception as e:
            print(f"[LLM] local tokenizer missing -> downloading… ({type(e).__name__})")
            tokenizer = BertTokenizer.from_pretrained(hf_id, local_files_only=False)
            
    else:
        raise Exception('LLM model is not defined')
    return llm_model, tokenizer