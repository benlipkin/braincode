import logging
import sys
from parser import CLI

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    CLI().run_main()
