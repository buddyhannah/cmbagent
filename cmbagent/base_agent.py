import os 
import logging
from cobaya.yaml import yaml_load_file
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.agentchat import UserProxyAgent

from cmbagent.utils import file_search_max_num_results
from autogen.agentchat import ConversableAgent, UpdateSystemMessage
import autogen
import copy
from autogen.agentchat import UserProxyAgent
from typing import Dict
import pandas as pd
# cmbagent_debug=True

cmbagent_debug = autogen.cmbagent_utils.cmbagent_debug

class CmbAgentUserProxyAgent(UserProxyAgent): ### this is for admin and executor 
    """A custom proxy agent for the user with redefined default descriptions."""

    # Override the default descriptions
    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task and provide feedback.", # default for admin 
        "TERMINATE": "A user that can run Python code and report back the execution results.",
        "NEVER": "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks).", # default for executor 
    }


class BaseAgent:

    def __init__(self, 
                 llm_config=None,
                 agent_id=None,
                 work_dir=None,
                 agent_type=None,
                 **kwargs):
        
        self.kwargs = kwargs

        if cmbagent_debug:
            print('\n\n in base_agent.py: __init__: llm_config: ', llm_config)
            print('\n\n')

        self.llm_config = copy.deepcopy(llm_config)
        #print(f"[DEBUG] llm_config {self.llm_config}")

        self.info = yaml_load_file(agent_id + ".yaml")

        self.name = self.info["name"]

        # if self.name == 'idea_maker':
        #     print('idea_maker: ', self.info)
        #     print('llm_config: ', self.llm_config)
        if len(self.llm_config['config_list']) > 0 and 'temperature' in self.llm_config['config_list'][0]:
            temperature = self.llm_config['config_list'][0]['temperature']
            self.llm_config['config_list'][0].pop('temperature')
            self.llm_config['temperature'] = temperature
            # print('llm_config: ', self.llm_config)

            # import sys; sys.exit()

        self.work_dir = work_dir

        self.agent_type = agent_type
        
        if cmbagent_debug:
            print('\n---------------------------------- setting name: ', self.info["name"])
            print('work_dir: ', self.work_dir)
            print('\n----------------------------------')


    def _debug_print_message_flow(self, messages):
        """Helper to print message flow for debugging"""
        print("\n=== MESSAGE FLOW DEBUG ===")
        for i, msg in enumerate(messages):
            print(f"{i}: {msg.get('role', 'no-role')} | {msg.get('content', 'no-content')[:100]}...")
            if 'tool_calls' in msg:
                print(f"    Tool calls: {msg['tool_calls']}")
            if 'tool_responses' in msg:
                print(f"    Tool responses: {msg['tool_responses']}")
        print("=======================\n")
        
 

        
    def _setup_native_retriever(self):
        try:
            from .retriever import VectorRetriever
            self.retriever = VectorRetriever()
            assert self.retriever is not None, "Retriever initialization failed!"
            self.docs = self.load_agent_documents(self.name)
            self.retriever.add_documents(self.docs)
              
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {str(e)}")  


    def load_agent_documents(self, agent_name):
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
        print(f"Loaded {len(docs)} documents for {agent_name}")
        return docs

    def set_rag_assistant_agent(self, **kwargs):
        """ Mistral RAG agent setup"""
        
        # Setup retriever 
        self._setup_native_retriever()
            
           
        # List files in the data_path excluding unwanted files 
        dir_path = os.getenv('CMBAGENT_DATA')
        data_path = os.path.join(dir_path, 'data', self.name.replace('_agent', ''))
        files = [f for f in os.listdir(data_path) if not (f.startswith('.') or f.endswith('.ipynb') or f.endswith('.yaml') or f.endswith('.txt') or os.path.isdir(os.path.join(data_path, f)))]

        if cmbagent_debug:
            print('\n\n\n\nin base_agent.py set_agent')
            print('files: ',files)
            # import sys; sys.exit()
            print("\n adding files to instructions: ", files)

        self.info["instructions"] += f'\n You have access to the following files: {files}.\n'
      
        
        # Create  agent 
        self.agent = CmbAgentSwarmAgent(
            name=self.name,
            system_message= self.info["instructions"],
            update_agent_state_before_reply=[UpdateSystemMessage(self.info["instructions"])],
            description=self.info["description"],
            llm_config={
                "config_list": self.llm_config["config_list"],
                "timeout": self.llm_config.get("timeout", 120),
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "file_search",
                        "description": "Searches and retrieves the most relevant documents for answering the given question.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                            "required": ["query"]
                        }
                    }
                }]
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            cmbagent_debug=cmbagent_debug,
        
        )
        
        # Register file_search
        self.agent.register_function(
            function_map={
                "file_search": self.file_search,
            }
        )

        print(f"\n=== AFTER AGENT CREATION ===")
        print(f"Handoffs object: {self.agent.handoffs}")
        print(f"Current after_work: {self.agent.handoffs.after_work}")
        
        
    
    def file_search(self, query: str):
        print(f"\n=== ENTERING file_search ===")
        #print(f"File search called with: {query}")
        # print(f"\nCurrent agent state:")
        # print(f"Handoffs object: {self.agent.handoffs}")
        # print(f"Current after_work: {self.agent.handoffs.after_work}")
        #if hasattr(self.agent, 'chat_messages'):
        #    self._debug_print_message_flow(self.agent.chat_messages[self.agent])
        
        if not hasattr(self, 'retriever') or self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call _setup_native_retriever() first.")
        results = self.retriever.search(query, file_search_max_num_results)
        return {
            "raw_results": results,  
            "query": query
        }



    

    ## for engineer/.. all non rag agents
    def set_assistant_agent(self,
                            instructions=None, 
                            description=None):
        
        if cmbagent_debug:
            print('\n\n\n\nin base_agent.py set_assistant_agent')
            print('name: ',self.name)
            # import sys; sys.exit()  

        if instructions is not None:

            self.info["instructions"] = instructions

        if description is not None:

            self.info["description"] = description

        logger = logging.getLogger(self.name) 
        logger.info("Loaded assistant info:")
        for key, value in self.info.items():
            logger.info(f"{key}: {value}")

        # print('setting assistant agent: ',self.name)
        # print('self.agent_type: ',self.agent_type)

        # if self.name == 'plan_setter':
        #     functions = [record_plan_constraints]
        # else:
        #     functions = []

        functions = []

        if self.name == 'cmbagent_tool_executor':
            self.agent = ConversableAgent(
                        name="cmbagent_tool_executor",
                        human_input_mode="NEVER",
                        llm_config=self.llm_config,
                    )

        else:
            self.agent = CmbAgentSwarmAgent(
                name=self.name,
                # system_message=self.info["instructions"],
                update_agent_state_before_reply=[UpdateSystemMessage(self.info["instructions"]),],
                description=self.info["description"],
                llm_config=self.llm_config,
                cmbagent_debug=cmbagent_debug,
                functions=functions,
            )
        


        if cmbagent_debug:
            print("AssistantAgent set.... moving on.\n")

    def set_code_agent(self,instructions=None):

        if instructions is not None:
            self.info["instructions"] = instructions

        logger = logging.getLogger(self.name) 
        logger.info("Loaded assistant info:")
        for key, value in self.info.items():
            logger.info(f"{key}: {value}")

        execution_policies = {
            "python": True,
            "bash": False,
            "shell": False,
            "sh": False,
            "pwsh": False,
            "powershell": False,
            "ps1": False,
            "javascript": False,
            "html": False,
            "css": False,
            }

        if 'bash' in self.name:
            execution_policies = {
                "python": False,
                "bash": True,
                "shell": False,
                "sh": False,
                "pwsh": False,
                "powershell": False,
                "ps1": False,
                "javascript": False,
                "html": False,
                "css": False,
            }

        self.agent = CmbAgentSwarmAgent(
            name= self.name,
            system_message= self.info["instructions"],
            description=self.info["description"],
            llm_config=self.llm_config,
            human_input_mode=self.info["human_input_mode"],
        max_consecutive_auto_reply=self.info["max_consecutive_auto_reply"],
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(work_dir=self.work_dir,
                                                    timeout=self.info["timeout"],
                                                    execution_policies = execution_policies
                                                    ),
            "last_n_messages": 2,
        },
        cmbagent_debug=cmbagent_debug,
        )

        if cmbagent_debug:
            print('code_agent set with work_dir: ', self.work_dir, '.... moving on.\n')


    def set_admin_agent(self,instructions=None):

        logger = logging.getLogger(self.name) 
        logger.info("Loaded assistant info:")

        for key, value in self.info.items():

            logger.info(f"{key}: {value}")

        self.agent = CmbAgentUserProxyAgent(
            name= self.name,
            update_agent_state_before_reply=[UpdateSystemMessage(self.info["instructions"]),],
            # system_message= self.info["instructions"],
            code_execution_config=self.info["code_execution_config"],
        )


class CmbAgentSwarmAgent(ConversableAgent):
    """CMB Swarm agent for participating in a swarm.

    CmbAgentSwarmAgent is a subclass of SwarmAgent, which is a subclass of ConversableAgent.

    Additional args:
        functions (List[Callable]): A list of functions to register with the agent.
    """
    pass