# cmbagent/utils.py
import os
import autogen
import pickle
import logging
from ruamel.yaml import YAML
from autogen.cmbagent_utils import cmbagent_debug


from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

cmbagent_debug = autogen.cmbagent_debug


# Get the path of the current file
path_to_basedir = os.path.dirname(os.path.abspath(__file__))
if cmbagent_debug:
    print('path_to_basedir: ', path_to_basedir)

# Construct the path to the APIs directory
path_to_apis = os.path.join(path_to_basedir, "apis")
if cmbagent_debug:
    print('path_to_apis: ', path_to_apis)

# Construct the path to the assistants directory
path_to_assistants = os.path.join(path_to_basedir, "agents/rag_agents/")
if cmbagent_debug:
    print('path_to_assistants: ', path_to_assistants)
path_to_agents = os.path.join(path_to_basedir, "agents/")

# Work directory
if "site-packages" in path_to_basedir or "dist-packages" in path_to_basedir:
    work_dir_default = os.path.join(os.getcwd(), "cmbagent_output")
    os.makedirs(work_dir_default, exist_ok=True)
else:
    work_dir_default = os.path.join(path_to_basedir, "../output")

if cmbagent_debug:
    print('\n\n\n\n\nwork_dir_default: ', work_dir_default)


default_chunking_strategy = {
    "type": "static",
    "static": {
        "max_chunk_size_tokens": 200, # reduce size to ensure better context integrity
        "chunk_overlap_tokens": 100 # increase overlap to maintain context across chunks
    }
}

# notes from https://platform.openai.com/docs/assistants/tools/file-search:

# max_chunk_size_tokens must be between 100 and 4096 inclusive.
# chunk_overlap_tokens must be non-negative and should not exceed max_chunk_size_tokens / 2.

# By default, the file_search tool outputs up to 20 chunks for gpt-4* models and up to 5 chunks for gpt-3.5-turbo. 
# You can adjust this by setting file_search.max_num_results in the tool when creating the assistant or the run.

default_top_p = 0.05
default_temperature = 0.00001


default_select_speaker_prompt_template = """
Read the above conversation. Then select the next role from {agentlist} to play. Only return the role.
Note that only planner can modify or update the PLAN. planner should not be selected after the PLAN has been approved.
executor should not be selected unless admin says "execute".
engineer should be selected to check for conflicts. 
engineer should be selected to check code. 
engineer should be selected to provide code to save summary of session. 
executor should be selected to execute. 
planner should be the first agent to speak. 
"""
### note that we hardcoded the requirement that planner speaks first. 


default_select_speaker_message_template = """
You are in a role play game about cosmological data analysis. The following roles are available:
                {roles}.
                Read the following conversation.
                Then select the next role from {agentlist} to play. Only return the role.
Note that only planner can modify or update the PLAN.
planner should not be selected after the PLAN has been approved.
executor should not be selected unless admin says "execute".
engineer should be selected to check for conflicts. 
engineer should be selected to check code. 
executor should be selected to execute. 
planner should be the first agent to speak.
"""


default_groupchat_intro_message = """
We have assembled a team of LLM agents and a human admin to solve Cosmological data analysis tasks. 

In attendance are:
"""

# TODO
# see https://github.com/openai/openai-python/blob/da48e4cac78d1d4ac749e2aa5cfd619fde1e6c68/src/openai/types/beta/file_search_tool.py#L20
# default_file_search_max_num_results = 20
# The default is 20 for `gpt-4*` models and 5 for `gpt-3.5-turbo`. This number
# should be between 1 and 50 inclusive.
file_search_max_num_results = autogen.file_search_max_num_results

default_max_round = 50

default_llm_model = "llama"

'''
Agents that use tool calling:
engineer, researcher, idea_maker, idea_hater, camb_context, classy_context, aas_keyword_finder
'''
default_agents_llm_model = {
    "engineer": "llama_tool",
    "aas_keyword_finder": "groq_tool",
    "task_improver": "exaone",
    "task_recorder": "llama_tool",
    "researcher": "groq_tool",
    "perplexity": "exaone",
    "planner": "qwen",
    "plan_reviewer": "groq",
    "idea_hater": "groq_tool",
    "idea_maker": "mistral_tool",
    
    # rag agents
    "classy_sz": "llama_tool",
    "camb": "mistral_tool",
    "classy": "llama_tool",
    "cobaya": "llama_tool",
    
    "planck": "llama_tool",
    
    "camb_context": "deepseek_tool",

    # HANNAH -ADDED
    "plan_setter": "groq_tool", # Uses tool call to run record_plan_constraints?
    "classy_context": "exaone", 
    "control": "deepseek_tool", # no response. Call record_status tool
    "control_starter": "deepseek_tool", # no response. Call record_status tool
    "engineer_nest": "llama", # trigger nested chat for engineer agent.
    "executor": "groq_tool", # ??????????????? Execution requires tool calling?
    "execute_bash": "mistral_tool", # execute code from installer
    "idea_hater_response_formatter": "llama", # format response from idea hater
    "idea_saver": "groq_tool", # no response. Call tool record_ideas 
    "plan_recorder":"deepseek_tool", # no response. Call tool record_plan
    "plot_judge":"groq_tool", # call_vlm_judge tool 
    "researcher_executor":"llama_tool", # save content provided by the researcher
    "review_recorder":"groq_tool",  # No response. Call tool record_review
    
    # formatting agents

    "classy_sz_response_formatter": "deepseek",
    "camb_response_formatter": "groq",
    "classy_response_formatter": "llama",
    "cobaya_response_formatter": "llama",
    "engineer_response_formatter": "deepseek",
    "researcher_response_formatter": "deepseek",
    "executor_response_formatter": "exaone",
    

   
}
'''
"classy_sz_response_formatter": "cloudflare",
    "camb_response_formatter": "cloudflare",
    "classy_response_formatter": "cloudflare",
    "cobaya_response_formatter": "cloudflare",
    "engineer_response_formatter": "cloudflare",
    "researcher_response_formatter": "cloudflare",
    "executor_response_formatter": "cloudflare",
    
'''

default_agent_llm_configs = {}

def get_api_keys_from_env():
    api_keys = {
        # Free apis
        "OPENROUTER": os.getenv("OPENROUTER_API_KEY"), # good
        "ARLIAI": os.getenv("ARLIAI_API_KEY"),
        "GROQ": os.getenv("GROQ_API_KEY"), # good
        "MISTRAL": os.getenv("MISTRAL_API_KEY"), 
        "LLAMA": os.getenv("LLAMA_API_KEY"), # good
        "TOGETHERAI": os.getenv("TOGETHERAI_API_KEY"), # good
        "CLOUDFLARE": os.getenv("CLOUDFLARE_API_KEY"),
        "CLOUDFLARE_ACCOUNT_ID": os.getenv("CLOUDFLARE_ACCOUNT_ID")
    }
    return api_keys

#Test cloudfare and Gemini 2.5 Pro 

def get_model_config(model, api_keys=None):
    """Returns a list of ModelClient instances with fallback options"""
    if api_keys is None:
        api_keys = get_api_keys_from_env()
    
    all_configs = {
        # require tool calling
        "mistral_tool": {
            "model": "mistral-small-latest",
            "api_key": api_keys.get("MISTRAL"),  
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "tool_choice": "any", 
            "top_p": default_top_p
        },
        "deepseek_tool": {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "api_key": api_keys["OPENROUTER"],
            "api_type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "tool_choice": "required",
            "top_p": default_top_p
        },
        "groq_tool": {
            "model": "llama-3.1-8b-instant",
            "api_key": api_keys["GROQ"],
            "api_type": "groq",
            "base_url": "https://api.groq.com",
            "tool_choice": "required",
            "top_p": default_top_p
        },
        "llama_tool": {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "api_key": api_keys.get("LLAMA"),
            "api_type": "openai",
            "base_url": "https://api.llama.com/compat/v1/",
            "tool_choice":"required",
            "top_p": default_top_p
        },
        
        "mistral": {
            "model": "mistral-small",
            "api_key": api_keys.get("MISTRAL"), 
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "tool_choice": "none", 
            "top_p": default_top_p
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
        "llama": {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "api_key": api_keys.get("LLAMA"),
            "api_type": "openai",
            "base_url": "https://api.llama.com/compat/v1/",
            "tool_choice":"none",
            "top_p": default_top_p
        },
        "exaone": {
            "model": "lgai/exaone-deep-32b",
            "api_key": api_keys.get("TOGETHERAI"),
            "api_type": "together",
            "base_url": "https://api.together.xyz/v1",
            "top_p": default_top_p,
            
        },
        "qwen": {
            "model": "Qwen3-14B",
            "api_key": api_keys.get("ARLIAI"),
            "api_type": "openai",
            "base_url": "https://api.arliai.com/v1",
            "tool_choice": "none",
            "top_p": default_top_p
        },
        "cloudflare": {
            "model": "@cf/meta/llama-3.1-8b-instruct-fast", #"@hf/thebloke/deepseek-coder-6.7b-instruct-awq",  
            "api_key": api_keys.get("CLOUDFLARE"),
            "api_type": "openai",
            "base_url": f"https://api.cloudflare.com/client/v4/accounts/{api_keys.get('CLOUDFLARE_ACCOUNT_ID')}/ai/v1",
            # "response_format": { "type": "json_object" }, 
        },

    }


    hasTool = "_tool" if  "tool" in model else '' 
    if "groq" in model:
        configs = [all_configs[f"groq{hasTool}"], all_configs[f"llama{hasTool}"], all_configs[f"mistral{hasTool}"], all_configs[f"deepseek{hasTool}"]]
    elif "mistral" in model:
        configs = [all_configs[f"mistral{hasTool}"], all_configs[f"deepseek{hasTool}"], all_configs[f"groq{hasTool}"], all_configs[f"llama{hasTool}"]]
    elif "llama" in model:
        configs = [all_configs[f"llama{hasTool}"], all_configs[f"mistral{hasTool}"], all_configs[f"deepseek{hasTool}"], all_configs[f"groq{hasTool}"]]
    elif "deepseek" in model:
        configs = [all_configs[f"deepseek{hasTool}"], all_configs[f"groq{hasTool}"], all_configs[f"llama{hasTool}"], all_configs[f"mistral{hasTool}"]]
    elif "cloudflare" in model:
        configs=[all_configs[f"cloudflare{hasTool}"]]
    # no tool calling
    elif "exaone" in model:
        configs = [all_configs["exaone"], all_configs["llama"], all_configs["groq"], all_configs[f"cloudflare"], all_configs["deepseek"],  all_configs["mistral"], all_configs["qwen"]]
    elif "qwen" in model:
        configs = [all_configs["qwen"], all_configs["exaone"], all_configs["llama"], all_configs["deepseek"],  all_configs["groq"], all_configs["cloudflare"], all_configs["mistral"]]
    else:
        raise ValueError(f"Invalid model {model}")
        
    return configs

# Initialize default_agent_llm_configs with list format
api_keys_env = get_api_keys_from_env()
default_agent_llm_configs = {
    agent: get_model_config(model, api_keys_env)
    for agent, model in default_agents_llm_model.items()
}

# Initialize default_llm_config_list with list format
default_llm_config_list = get_model_config(default_llm_model, api_keys_env)



#### note we should be able to set the temperature for different agents, e.g., 
                    # "idea_maker": {
                    #     "model": default_llm_model,
                    #     "api_key": os.getenv("OPENAI_API_KEY"),
                    #     "api_type": "openai",
                    #     'temperature': 0.5,
                    #     },



def update_yaml_preserving_format(yaml_file, agent_name, new_id, field = 'vector_store_ids'):
    yaml = YAML()
    yaml.preserve_quotes = True  # This preserves quotes in the YAML file if they are present

    # Load the YAML file while preserving formatting
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.load(file)
    
    # Update the vector_store_id for the specific agent
    if yaml_content['name'] == agent_name:
        if field == 'vector_store_ids':
            yaml_content['assistant_config']['tool_resources']['file_search']['vector_store_ids'][0] = new_id
        elif field == 'assistant_id':
            yaml_content['assistant_config']['assistant_id'] = new_id
    else:
        print(f"Agent {agent_name} not found.")
    
    # Write the changes back to the YAML file while preserving formatting
    with open(yaml_file, 'w') as file:
        yaml.dump(yaml_content, file)

def aas_keyword_to_url(keyword):
    """
    Given an AAS keyword, return its IAU Thesaurus URL.
    
    Args:
        keyword (str): The AAS keyword (e.g., "H II regions")
        
    Returns:
        str: The corresponding IAU Thesaurus URL
    """
    with open('aas_kwd_to_url.pkl', 'rb') as f:
        dic = pickle.load(f)
    return dic[keyword]


with open(path_to_basedir + '/aas_kwd_to_url.pkl', 'rb') as file:
    AAS_keywords_dict = pickle.load(file)

# print(my_dict)
# Assuming you have already loaded your dictionary into `my_dict`
AAS_keywords_string = ', '.join(AAS_keywords_dict.keys())


camb_context_url = "https://camb.readthedocs.io/en/latest/_static/camb_docs_combined.md"
classy_context_url = "https://github.com/santiagocasas/clapp/tree/main/classy_docs.md"

