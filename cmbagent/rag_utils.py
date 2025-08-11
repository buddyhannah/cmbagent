import os
import importlib
from openai import OpenAI
from autogen.cmbagent_utils import cmbagent_debug
import requests
import pprint
from .utils import path_to_assistants,default_chunking_strategy,YAML,update_yaml_preserving_format
from .retriever import VectorRetriever # Hannah added


def import_rag_agents():        
    imported_rag_agents = {}
    for filename in os.listdir(path_to_assistants):
        if filename.endswith(".py") and filename != "__init__.py" and filename[0] != ".":
            module_name = filename[:-3]  # Remove the .py extension
            class_name = ''.join([part.capitalize() for part in module_name.split('_')]) + 'Agent'
            module_path = f"cmbagent.agents.rag_agents.{module_name}"
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            imported_rag_agents[class_name] = {}
            imported_rag_agents[class_name]['agent_class'] = agent_class
            imported_rag_agents[class_name]['agent_name'] = module_name
    return imported_rag_agents




# Hannah modified
def push_vector_stores(cmbagent_instance, make_vector_stores, chunking_strategy, verbose = False):
    if not make_vector_stores:
        return
    
    for agent in cmbagent_instance.agents:
        if agent.name not in cmbagent_instance.non_rag_agent_names:
            if verbose:
                print(f"Setting up retriever for {agent.name}")
            
            # Initialize retriever with agent-specific config
            if not hasattr(agent, 'retriever'):
                from .retriever import VectorRetriever
                agent.retriever = VectorRetriever()
                
                # Load documents
                docs = load_agent_documents(agent.name)
                agent.retriever.add_documents(docs)
                
                if verbose:
                    print(f"Loaded {len(docs)} documents for {agent.name}")


def make_rag_agents(make_new_rag_agents):
    """
    Create new RAG agents based on the provided list of agent names.

    This function generates Python and YAML files for each new agent specified
    in the 'make_new_rag_agents' list. It creates:
    1. A Python file with a basic agent class structure.
    2. A YAML file with initial configuration for the agent.
    3. A data folder for each agent to store relevant files.

    Args:
        make_new_rag_agents (list): A list of strings, where each string is the
                                    name of a new agent to be created.

    Returns:
        dict: A dictionary where keys are agent names and values are paths to
              their respective data folders.

    Note:
    - The Python file will contain a class definition inheriting from BaseAgent.
    - The YAML file will include basic configuration like name, instructions,
      and tool definitions.
    - Existing files with the same names will be overwritten.
    - A new data folder is created for each agent in the assistants directory.
    """
    data_folders = {}
    for agent_name in make_new_rag_agents:
        # Create the Python file for the agent
        agent_file_path = os.path.join(path_to_assistants, f"{agent_name}.py")
        with open(agent_file_path, "w") as f:
            f.write(f"""import os
from cmbagent.base_agent import BaseAgent


class {agent_name.capitalize()}Agent(BaseAgent):

    def __init__(self, llm_config=None, **kwargs):

        agent_id = os.path.splitext(os.path.abspath(__file__))[0]

        super().__init__(llm_config=llm_config, agent_id=agent_id, **kwargs)
""")

        # Create the YAML file for the agent
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        
        yaml_file_path = os.path.join(path_to_assistants, f"{agent_name}.yaml")


        yaml_content = {
            "name": f"{agent_name}_agent",
            "instructions": f"You are the {agent_name}_agent in the team. Your role is to assist with tasks related to {agent_name}.",
            "description": f"This is the {agent_name}_agent: a retrieval agent that provides assistance with {agent_name.upper()}. It must perform retrieval augmented generation and include the <filenames> in the response.",
            "allowed_transitions": ["admin"],
            
        }
        
        with open(yaml_file_path, "w") as f:
            yaml.dump(yaml_content, f)

        print(f"Created {agent_name} agent files: {agent_file_path} and {yaml_file_path}")
        # Create a folder for the agent's data
        # agent_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', agent_name)
        dir_path = os.getenv('CMBAGENT_DATA')
        agent_data_folder = os.path.join(dir_path, 'data', agent_name)
        print(f"Creating data folder for {agent_name} agent: {agent_data_folder}")
        os.makedirs(agent_data_folder, exist_ok=True)
        print(f"Created data folder for {agent_name} agent: {agent_data_folder}")
        print(f"Please deposit any relevant files for the {agent_name} agent in this folder.")

    # Return a dictionary with the full paths to the agent data folders
    data_folders = {}
    # data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    data_dir = os.path.join(dir_path, 'data')
    for agent_folder in os.listdir(data_dir):
        full_path = os.path.join(data_dir, agent_folder)
        if os.path.isdir(full_path):
            data_folders[agent_folder] = full_path
    return data_folders


  
def load_agent_documents(agent_name):
    """Load documents for an agent from cmbagent_data"""
    docs = []
    data_dir = os.path.join(os.getenv('CMBAGENT_DATA'), 'data', agent_name.replace('_agent', ''))
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith('.'):
                continue
                
            filepath = os.path.join(data_dir, filename)
            try:
                if filename.endswith('.md'):
                    with open(filepath, 'r') as f:
                        docs.append(f.read())
                elif filename.endswith('.pdf'):
                    from .retriever import VectorRetriever
                    docs.append(VectorRetriever.pdf_to_text(filepath))
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    print(f"docs length: {len(docs)}")
    return docs
