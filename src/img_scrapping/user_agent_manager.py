import random

import requests
from omegaconf import OmegaConf


class UserAgentManager:
    def __init__(self, cfg_user_agent_manager):
        self.user_agents = cfg_user_agent_manager.user_agents
        self.session = requests.Session()
        self.headers = cfg_user_agent_manager.headers
        self.rotate_user_agent()

    def rotate_user_agent(self):
        headers = OmegaConf.to_container(self.headers, resolve=True)
        headers["User-Agent"] = random.choice(self.user_agents)
        self.session.headers.update(headers)

    def get_session(self):
        return self.session
