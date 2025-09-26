
import os
import importlib
import json
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern

from .utils import work_dir_default

from .utils import (get_model_config, default_temperature, default_agents_llm_model, get_api_keys_from_env)

from .namac_functions import register_nuclear_functions


from typing import Dict, List, Literal
from autogen import Agent
from autogen.agentchat.group import ReplyResult, AgentTarget, Handoffs, TerminateTarget
from autogen.agentchat.contrib.capabilities import transforms
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter


class NuclearAgent:
    """ For defining the nuclear reactor control agent"""
    
    def __init__(
                self,
                reactor_state_file: str = "nuclear_plant.csv",
                agent_description_json: str = "namac_agent_descriptions.json",
                agent_configs: Dict = None,
                work_dir: str = "./output/namac",
                agents_base_path: str = "/home/dell/cmbagent/namac/agents/",
                is_planning_and_control: bool = False):
        """
        Args:
            reactor_state_file: Path to CSV reactor state file
            agent_description_json: Path to the agent domain descriptions
            agent_configs: agent LLM configurations
            work_dir: Working directory for logs/outputs
            agents_base_path: path for
        """
        
        self.agents_base_path = agents_base_path
        self.reactor_state_file = reactor_state_file
        self.work_dir = work_dir
        
        self.is_planning_and_control = is_planning_and_control
        self.agent_description_json = agent_description_json
        
        # Create working directory
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(f"{self.work_dir}/chats", exist_ok=True)
        
        # Initialize agents
        self.agent_classes = self.import_agent_classes()
        self.llm_config = self.create_llm_configs(agent_configs)
        self.agents = self.initialize_agents(self.llm_config)
        
        
        self._register_handoffs()
        register_nuclear_functions(self)

    def create_llm_configs(self, agent_configs: Dict) -> Dict:
        """Create the config list"""
        prepared_configs = {}
        for agent_name in self.agent_classes.keys():
            agent_config = {
                "temperature": default_temperature,
                "timeout": 1200,
                "config_list": agent_configs.get(agent_name, [])  # Fallback to empty list for admin
            }
            prepared_configs[agent_name] = agent_config
            
        return prepared_configs


        
    def import_agent_classes(self) -> Dict:
        """Import all agent classes from their modules"""
        
        agent_dirs = [
            'diagnosis',
            'strategy_inventory', 
            'prognosis',
            'strategy_assessment',
            'updater',
            'updater_helper',
            'planner_helper'
        ]
        
        if self.is_planning_and_control:
            agent_dirs += ['planner', 'control', 'summarizer', 'admin', 'scenario_builder', 'planner_helper']
        
        agent_classes = {}
        for agent_dir in agent_dirs:
            agent_path = os.path.join(self.agents_base_path, agent_dir)
            
            # Find.py file in  agent directory
            for filename in os.listdir(agent_path):
                if filename.endswith(".py") and filename != "__init__.py" and not filename.startswith("."):
                    module_name = filename[:-3]  # Remove .py extension
                    class_name = ''.join([part.capitalize() for part in module_name.split('_')]) + 'Agent'
                    
                    # Import the module
                    module_path = f"namac.agents.{agent_dir}.{module_name}"
                    try:
                        module = importlib.import_module(module_path)
                        agent_class = getattr(module, class_name)
                        agent_classes[agent_dir] = agent_class
                    except (ImportError, AttributeError) as e:
                        print(f"Error loading agent {agent_dir}: {e}")
                        raise
                        
        return agent_classes
    


    # https://microsoft.github.io/autogen/0.2/docs/topics/handling_long_contexts/intro_to_transform_messages/
    def initialize_agents(self, configs: Dict) -> Dict[str, Agent]:
        agents = {}
        
        # TODO Limit message history
        message_limits = {
            "planner": 2,
            "control": 5,
            
            "diagnosis": 3,
            "strategy_inventory": 3,
            "prognosis": 5,
            "strategy_assessment": 5,
            "updater": 5,
            "summarizer": 10
        }
       

        
        
        for agent_name, agent_class in self.agent_classes.items():
           
            # Get system message
            yaml_path = os.path.join(self.agents_base_path, agent_name, f"{agent_name}.yaml")
            with open(yaml_path, 'r') as f:
                system_message = f.read()
            
            
            agent = agent_class(
                name=agent_name,
                system_message=system_message,
                llm_config=configs.get(agent_name),
                agent_type='swarm', 
                work_dir=self.work_dir,
                is_termination_msg=lambda x: False
            )
            
            # Initialize handoffs 
            if not hasattr(agent, 'handoffs'):
                agent.handoffs = Handoffs()
    
            # Call set_agent
            agent.set_agent()
            
            '''
            lookback_limit = message_limits.get(agent_name, 5)
            limiter = MessageHistoryLimiter(max_messages=lookback_limit, keep_first_message=False)
            transformer = TransformMessages(transforms=[limiter], verbose=False)
            transformer.add_to_agent(agent.agent)
            '''
                
            agents[agent_name] = agent
   
        return agents

    
    def _register_handoffs(self):
        
        # Planning and control mode: transfer to Control after each step
        agents = self.agents
        if self.is_planning_and_control:
            for agent_name, agent in agents.items():
                if agent_name == "control":
                    continue
                elif agent_name == "admin":
                    agent.agent.handoffs.set_after_work(AgentTarget(agents['planner'].agent))
                elif agent_name == "updater_helper":
                    agent.agent.handoffs.set_after_work(AgentTarget(agents['updater'].agent))
                elif agent_name == "planner":
                    agent.agent.handoffs.set_after_work(AgentTarget(agents['planner_helper'].agent))
                else:
                    agent.agent.handoffs.set_after_work(AgentTarget(agents['control'].agent))
        
        # Hardcode order of agent execution
        else:
            agents['diagnosis'].agent.handoffs.set_after_work(AgentTarget(agents['strategy_inventory'].agent))
            agents['strategy_inventory'].agent.handoffs.set_after_work(AgentTarget(agents['prognosis'].agent))
            agents['prognosis'].agent.handoffs.set_after_work(AgentTarget(agents['strategy_assessment'].agent))
            agents['strategy_assessment'].agent.handoffs.set_after_work(AgentTarget(agents['updater_helper'].agent))
            agents['updater_helper'].agent.handoffs.set_after_work(AgentTarget(agents['updater'].agent))
            agents['updater'].agent.handoffs.set_after_work(AgentTarget(agents['diagnosis'].agent))
             
    
    
    def run_cycle(self, max_cycles: int = 10, user_query: str = None):
        """Start the chat"""
        from autogen.agentchat import initiate_group_chat
        
        with open(self.agent_description_json, "r") as f:
            agent_data = json.load(f)


        # Prepare context
        main_task = user_query if user_query else "Get the nuclear reactor state and identify the SSFs"
        context = ContextVariables(data={
            'agent_descriptions': agent_data, 
            'main_task': main_task,
            'work_dir': self.work_dir,
        })

        # Get the agents
        autogen_agents = [agent.agent for agent in self.agents.values()]
        
        if self.is_planning_and_control:
            initial_agent = self.agents['planner'].agent
        else:
            initial_agent = self.agents['diagnosis'].agent
        
    
        # Create agent pattern
        start_agent =  self.llm_config['planner'] if self.is_planning_and_control else  self.llm_config['diagnosis']
        agent_pattern = AutoPattern(
            agents=autogen_agents,
            initial_agent=initial_agent,  
            context_variables=context,
            group_manager_args={
                "llm_config": start_agent,
                "name": "nuclear_control"
            }
        )
        
        # 3. Start chat
        chat_result, context_variables, last_agent = initiate_group_chat(
            pattern=agent_pattern,
            messages=context['main_task'],
            max_rounds=max_cycles * len(self.agents)
        )
    
        return chat_result


def namac_planning_and_control(task,
                         reactor_state_file,
                         work_dir=work_dir_default,
                         max_rounds=50,
                         api_keys=None):
    """Nuclear plant control loop"""
    api_keys = api_keys or get_api_keys_from_env()
    
    nuclear_agent = NuclearAgent(
        reactor_state_file=reactor_state_file,
        agent_configs={
            'diagnosis': get_model_config(default_agents_llm_model['diagnosis'], api_keys),
            'strategy_inventory': get_model_config(default_agents_llm_model['strategy_inventory'], api_keys),
            'prognosis': get_model_config(default_agents_llm_model['prognosis'], api_keys),
            'strategy_assessment': get_model_config(default_agents_llm_model['strategy_assessment'], api_keys),
            'updater': get_model_config(default_agents_llm_model['updater'], api_keys),
            'control': get_model_config(default_agents_llm_model['namac_control'], api_keys),
            'planner': get_model_config(default_agents_llm_model['namac_planner'], api_keys),
            'summarizer': get_model_config(default_agents_llm_model['namac_summarizer'], api_keys),
            'scenario_builder': get_model_config(default_agents_llm_model['scenario_builder'], api_keys),
        },
        work_dir=work_dir,
        is_planning_and_control = True
    )
    
    return nuclear_agent.run_cycle(max_rounds, task)



def namac_hardcoded(reactor_state_file,
                         work_dir=work_dir_default,
                         max_rounds=50,
                         api_keys=None):
    """Nuclear plant control loop"""
    api_keys = api_keys or get_api_keys_from_env()
    
    nuclear_agent = NuclearAgent(
        reactor_state_file=reactor_state_file,
        agent_configs={
            'diagnosis': get_model_config(default_agents_llm_model['diagnosis'], api_keys),
            'strategy_inventory': get_model_config(default_agents_llm_model['strategy_inventory'], api_keys),
            'prognosis': get_model_config(default_agents_llm_model['prognosis'], api_keys),
            'strategy_assessment': get_model_config(default_agents_llm_model['strategy_assessment'], api_keys),
            'updater': get_model_config(default_agents_llm_model['updater'], api_keys),
        },
        work_dir=work_dir
    )
    
    return nuclear_agent.run_cycle(max_rounds)