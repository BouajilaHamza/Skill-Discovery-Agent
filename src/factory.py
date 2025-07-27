from typing import Dict,Any,Type
from src.agents.base_agent import BaseAgent
from src.agents.diayn_agent import DIAYNAgent



AGENT_REGISTRY = {
    "diayn":DIAYNAgent
}



def create_agent(agent_type: str, config: Dict[str, Any]) -> BaseAgent:
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_cls = AGENT_REGISTRY[agent_type]
    return agent_cls(config)