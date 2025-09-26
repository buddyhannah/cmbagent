# namac/utils.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

# Base paths
path_to_basedir = os.path.dirname(os.path.abspath(__file__))
path_to_agents = os.path.join(path_to_basedir, "agents")

# Work directory
if "site-packages" in path_to_basedir or "dist-packages" in path_to_basedir:
    work_dir_default = os.path.join(os.getcwd(), "namac_output")
    os.makedirs(work_dir_default, exist_ok=True)
else:
    work_dir_default = os.path.join(path_to_basedir, "../output")

# Default model parameters
default_top_p = 0.05
default_temperature = 0.00001
default_llm_model = "llama"

# Llama API does not have access to any execution environment!
default_agents_llm_model = {
    "namac_planner": "mistral_tool", # deepseek
    "namac_control": "mistral_tool",
    "diagnosis": "mistral_tool",
    "strategy_inventory": "llama",
    "prognosis": "llama",
    "strategy_assessment": "llama",
    "updater": "mistral_tool",
    "namac_summarizer": "llama",
    "scenario_builder": "mistral_tool",
}

def get_api_keys_from_env():
    return {
        "OPENROUTER": os.getenv("OPENROUTER_API_KEY"),
        "GROQ": os.getenv("GROQ_API_KEY"),
        "MISTRAL": os.getenv("MISTRAL_API_KEY"),
        "LLAMA": os.getenv("LLAMA_API_KEY"),
        "CLOUDFLARE": os.getenv("CLOUDFLARE_API_KEY"),
        "CLOUDFLARE_ACCOUNT_ID": os.getenv("CLOUDFLARE_ACCOUNT_ID")
    }

def get_model_config(model, api_keys=None):
    """Returns a list of ModelClient configs with fallback options"""
    if api_keys is None:
        api_keys = get_api_keys_from_env()

    all_configs = {
        "mistral_tool": {
            "model": "mistral-small-latest",
            "api_key": api_keys.get("MISTRAL"),
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "tool_choice": "any",
            "top_p": default_top_p,
        },
        "mistral1_tool": {
            "model": "mistral-medium-latest",
            "api_key": api_keys.get("MISTRAL"),
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "tool_choice": "any",
            "top_p": default_top_p,
        },
        "groq_tool": {
            "model": "llama-3.1-8b-instant",
            "api_key": api_keys.get("GROQ"),
            "api_type": "groq",
            "base_url": "https://api.groq.com",
            "tool_choice": "required",
            "top_p": default_top_p,
        },
        "deepseek_tool": {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "api_key": api_keys["OPENROUTER"],
            "api_type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "tool_choice": "required",
            "top_p": default_top_p
        },
        "cloudflare_tool": {
            "model": "@hf/nousresearch/hermes-2-pro-mistral-7b",   
            "api_key": api_keys.get("CLOUDFLARE"),
            "api_type": "openai",
            "base_url": f"https://api.cloudflare.com/client/v4/accounts/{api_keys.get('CLOUDFLARE_ACCOUNT_ID')}/ai/v1",
            "tool_choice": "required"

        },
        
        
        "mistral": {
            "model": "mistral-small",
            "api_key": api_keys.get("MISTRAL"), 
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "tool_choice": "none", 
            "top_p": default_top_p
        },
        "mistral1": {
            "model": "mistral-medium",
            "api_key": api_keys.get("MISTRAL"), 
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "tool_choice": "none", 
            "top_p": default_top_p
        },
        "llama": {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "api_key": api_keys.get("LLAMA"),
            "api_type": "openai",
            "base_url": "https://api.llama.com/compat/v1/",
            "tool_choice": "none",
            "top_p": default_top_p,
        },
        
        "deepseek": {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "api_key": api_keys["OPENROUTER"],
            "api_type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "tool_choice": "none",
            "top_p": default_top_p
        },
        "groq": {
            "model": "llama-3.1-8b-instant",
            "api_key": api_keys["GROQ"],
            "api_type": "groq",
            "base_url": "https://api.groq.com",
            "tool_choice": "none",
            "top_p": default_top_p
        },
        
       
    }
    
    
    hasTool = "_tool" if  "tool" in model else '' 
    if "llama" in model:
        configs = [all_configs["llama"], all_configs["deepseek"], all_configs["mistral"], all_configs[f"groq"]]
    elif "cloudflare_tool" in model:
        return [all_configs[model]]
    elif "groq" in model:
        configs = [all_configs[f"groq{hasTool}"], all_configs[f"cloudflare_tool"], all_configs[f"deepseek{hasTool}"], all_configs[f"mistral{hasTool}"], all_configs[f"mistral1{hasTool}"]]
    elif "mistral" in model:
        configs = [all_configs[f"mistral{hasTool}"], all_configs[f"mistral1{hasTool}"], all_configs[f"groq{hasTool}"], all_configs[f"cloudflare_tool"], all_configs[f"deepseek{hasTool}"]]
    elif "deepseek" in model:
        configs = [all_configs[f"deepseek{hasTool}"], all_configs[f"mistral{hasTool}"], all_configs[f"mistral1{hasTool}"], all_configs[f"groq{hasTool}"], all_configs[f"cloudflare_tool"]]
    else:
        raise ValueError(f"Invalid model {model}")
    


    return configs

# Initialize configs
api_keys_env = get_api_keys_from_env()
default_agent_llm_configs = {
    agent: get_model_config(model, api_keys_env)
    for agent, model in default_agents_llm_model.items()
}
default_llm_config_list = get_model_config(default_llm_model, api_keys_env)

