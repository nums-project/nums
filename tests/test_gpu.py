import nums
from nums import numpy as nps
from nums.core import settings
from nums.core.application_manager import instance

settings.backend_name = "gpu"

print(settings.backend_name)




def main():
    nums.init()



if __name__ == "__main__":
    main()