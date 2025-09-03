import os
from pydantic import BaseModel, Field
from cmbagent.base_agent import BaseAgent

class SummarizerAgent(BaseAgent):
    
    class SummaryCommand(BaseModel):
        summary: str = Field(..., description="Summary of previous iteration")
        
    def __init__(self, llm_config=None, **kwargs):

        agent_id = os.path.splitext(os.path.abspath(__file__))[0]

        super().__init__(llm_config=llm_config, agent_id=agent_id, **kwargs)


    def set_agent(self,**kwargs):

        self.llm_config['config_list'][0]['response_format'] = self.SummaryCommand
        super().set_assistant_agent(**kwargs)