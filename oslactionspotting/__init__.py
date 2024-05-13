__version_info__ = ('0', '1', '8')
__version__ = '.'.join(__version_info__)
__authors__ = ["Silvio Giancola", "Yassine Benzakour"]
__authors_username__ = "giancos"
__author_email__ = ["silvio.giancola@kaust.edu.sa", "yassine.benzakour@student.uliege.be"]
__github__ = 'https://github.com/OpenSportsLab/OSL-ActionSpotting'

from .apis import *
from .core import *
from .datasets import *
from .models import *