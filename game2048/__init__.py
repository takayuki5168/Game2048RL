import pkg_resources

__version__ = pkg_resources.get_distribution(
    'game2048-env').version

from game2048_env import Game2048Env
