import numpy as np
import fileinput
import re

def main():

    total = 0
    for line in fileinput.input():

        if line.startswith("#"):
            param_sz = re.findall("{(.+)}", line)[0]
            dims = [int(d) for d in param_sz.split(",")]
            total += np.prod(dims)

    print(total)


if __name__ == '__main__':
    main()
2
