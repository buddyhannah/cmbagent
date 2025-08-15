import sys
import os
import json

import autogen.cmbagent_utils
#autogen.cmbagent_utils.cmbagent_debug = True

from cmbagent import planning_and_control, one_shot, nuclear_plant_control, human_in_the_loop
from cmbagent.base_agent import BaseAgent
import cmbagent
from autogen.agentchat.group import AgentTarget
from autogen.agentchat import UserProxyAgent


def get_api_keys_from_env():
    """Retrieve API keys from environment variables."""
    return {
        "OPENROUTER": os.getenv("OPENROUTER_API_KEY"),
        "ARLIAI": os.getenv("ARLIAI_API_KEY"),
        "GROQ": os.getenv("GROQ_API_KEY"),
        "MISTRAL": os.getenv("MISTRAL_API_KEY"),
        "LLAMA": os.getenv("LLAMA_API_KEY"),
        "TOGETHERAI": os.getenv("TOGETHERAI_API_KEY"),
        "CLOUDFLARE": os.getenv("CLOUDFLARE_API_KEY"),
        "CLOUDFLARE_ACCOUNT_ID": os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    }

def camb_query():
    print("\n=== STARTING RAG AGENT TEST ===\n")

    # Initialize agents
    api_keys = get_api_keys_from_env()

    llm_config = {
        "config_list": [{
            "model": "mistral-small-latest",
            "api_type": "mistral",
            "api_key": api_keys["MISTRAL"],
            "base_url": "https://api.mistral.ai/v1"
        }],
        "timeout": 120,
        "tools": [{
            "type": "function",
            "function": {
                "name": "file_search",
                "description": "Search documentation and code files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }]
    }

    agent = cmbagent.CMBAgent(
        agent_list=['camb', 'classy_sz'],
        llm_config=llm_config,
        make_vector_stores=['camb', 'classy_sz'], # , 'cobaya', 'planck'
        verbose=True,
        skip_rag_agents=False
    )

    camb_agent = agent.get_agent_object_from_name('camb_agent')

    # Test tool
    # print("\n=== TESTING FILE_SEARCH TOOL ===")
    # tool_result = camb_agent.file_search("cosmological parameters")
    # print("Tool result:", tool_result if tool_result else "No results")

    # Test chat
    print("\n=== TESTING AGENT CHAT ===")
    task = (
        "What camb function should I use to calculate the reionization redshift given optical depth tau?"
    )

    chat_result = camb_agent.agent.initiate_chat(
        recipient=camb_agent.agent,
        message=task,
        #max_turns=3
    )
    print("\n=== FINAL CHAT RESULT ===")
    print(chat_result)
    
    
    
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <case_number>")
        sys.exit(1)

    try:
        case = int(sys.argv[1])
    except ValueError:
        print("case_number must be an integer.")
        sys.exit(1)

    if case == 1:
        task = open('prompts/prompt4.txt').read()
        results = planning_and_control(
            task=task,
            max_rounds_control=500,
            n_plan_reviews=1,
            max_n_attempts=4,
            max_plan_steps=7,
            plan_instructions=(
                "Use engineer agent for the whole analysis, and researcher at the very end "
                "in the last step to comment on results."
            )
        )
    
    elif case == 2:
        task = open('prompts/cambPrompt.txt').read()
        results = one_shot(task=task, max_n_attempts=5)

    elif case == 3:
        camb_query()
        
    elif case == 4:
        nuclear_plant_control(reactor_state_file="/home/dell/cmbagent/reactor_state.csv")
        
    elif case == 5:
        task = open('prompts/knapsackPrompt.txt').read()
        results = human_in_the_loop(task=task, agent="researcher")

        
  
    else:
        print("Invalid case number")
        sys.exit(1)



if __name__ == "__main__":
    main()
