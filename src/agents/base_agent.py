from abc import ABC,abstractmethod
import pytorch_lightning as pl
from typing import Dict, Any



class BaseAgent(pl.LightningModule, ABC):
    """Base class for agents"""
    
    def __init__(self,config:Dict[str,Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
    @abstractmethod
    def act(self,obs:Dict[str,Any]):
        """select an action given the observation"""
        pass
    
    @abstractmethod
    def training_step(self,batch:Dict[str,Any],batch_idx:int):
        """training step for the agent"""
        pass
        
    
    def configure_optimizers(self):
        """configure the optimizer for the agent"""
        raise NotImplementedError("Subclasses must implement the configure_optimizers method.")
    
    
    def save_checkpoint(self,path:str):
        """save the checkpoint for the agent"""
        self.trainer.save_checkpoint(path)
    
    @classmethod
    def load_checkpoint(cls,checkpoint_path:str,config=None):
        """load the checkpoint for the agent"""
        if config is None :
            return cls.load_from_checkpoint(checkpoint_path)
        return cls.load_from_checkpoint(checkpoint_path,config=config)
    
    
    