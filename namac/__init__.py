# import warnings
# warnings.filterwarnings("ignore", message=r'Field "model_client_cls" in LLMConfigEntry has conflict with protected namespace "model_"')

import warnings

warnings.filterwarnings(
    "ignore",                               # action
    message=r"Update function string contains no variables\.",  # regex
    category=UserWarning,                   # same category that’s raised
    module=r"autogen\.agentchat\.conversable_agent"  # where it comes from
)

from .namac import NuclearAgent

from .namac import namac_hardcoded, namac_planning_and_control
