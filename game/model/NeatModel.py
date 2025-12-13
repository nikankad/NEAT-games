# code for neat nn
class NeatModel:
    def __init__(self, config_path):
        """
        Initialize NEAT neural network model.
        
        Args:
            config_path (str): Path to NEAT configuration file
        """
        self.config_path = config_path
        self.net = None
        self.genome = None
        