import sys
import glob
import pandas

def main():
    fn = sys.argv[1]
    best = pandas.read_csv(fn)

    for _, row in best.iterrows():
        print(row['attention'])

        files = glob.glob(f"{row['fn_prefix']}*")
        print(files)



if __name__ == '__main__':
    main()

