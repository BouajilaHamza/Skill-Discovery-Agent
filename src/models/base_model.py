import torch.nn as nn




class BaseModel(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        
    