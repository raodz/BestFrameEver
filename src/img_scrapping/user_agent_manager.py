import random

import requests
from omegaconf import OmegaConf


class UserAgentManager:
    def __init__(self, cfg):
        self.user_agents = cfg.user_agents
        self.session = requests.Session()
        self.headers = cfg.headers
        self.rotate_user_agent()

    def rotate_user_agent(self):
        headers = OmegaConf.to_container(self.headers, resolve=True)
        headers["User-Agent"] = random.choice(self.user_agents)
        self.session.headers.update(headers)

    def get_session(self):
        return self.session
