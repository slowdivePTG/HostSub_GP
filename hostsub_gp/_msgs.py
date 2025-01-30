# hostsub_gp/_msgs.py
class msgs():
    """Class to print messages of multiple types."""

    # ANSI color codes
    GREEN = '\033[92m'      # Info
    RED = '\033[91m'        # Error
    YELLOW = '\033[93m'     # Warning
    BLUE = '\033[94m'       # Parameter
    BOLD = '\033[1m'        # Bold text
    RESET = '\033[0m'       # Reset color

    @staticmethod
    def info(message: str):
        print(f"{msgs.GREEN}{msgs.BOLD}[INFO]    :: {msgs.RESET}" + message)

    @staticmethod
    def error(message: str):
        print(f"{msgs.RED}{msgs.BOLD}[ERROR]   :: {msgs.RESET}" + message)

    @staticmethod
    def warning(message: str):
        print(f"{msgs.YELLOW}{msgs.BOLD}[WARNING] :: {msgs.RESET}" + message)

    @staticmethod
    def parameter(message: str):
        print(f"{msgs.BLUE}{msgs.BOLD}[PARAM]   :: {msgs.RESET}" + message)