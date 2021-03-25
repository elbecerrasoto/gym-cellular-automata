# Compiting forces
# Perception and Induction

# Selector ABC
# An abstract method is one that is declared but contains not implementation
# Cannot be instantiated and require subclaassing only

from abc import ABC, abstractmethod

class AbstractClassExample(ABC):
    def __init__(self, value):
        self.value = value
        super.__init__()
        
    @abstractmethod
    def do_something(self):
        pass

# A class that is derived from an abstract class cannot be instantiated
