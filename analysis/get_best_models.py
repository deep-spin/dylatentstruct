import sys
import os
import shutil
import glob
import pandas

def main():
    fn = sys.argv[1]
    best = pandas.read_csv(fn)

    print(best.columns)
    script_f = open("run_test.sh", "w")

    for _, row in best.iterrows():
        print(row['attention'])
        print(row['best_valid_acc'])

        files = sorted(list(glob.glob(f"{row['fn_prefix']}*{row['Run ID']}*")))

        if len(files):
            in_f = files[-1]
            print(in_f)
            out_f = sys.argv[2] + os.path.basename(in_f)
            shutil.copy(in_f, out_f)
            
            print(f"""
build/decomp --test \\
--saved-model {out_f} \\
--dataset {row['dataset']} \\
--dim 100 \\
--batch-size {row['batch_size']} \\
--normalize-embed \\
--drop 0 \\
--attn {row['attention']} \\
--sparsemap-eta {row['SM_eta']} \\
--sparsemap-max-iter {row['SM_maxit']} \\
--sparsemap-active-set-iter {row['SM_ASET_maxit']} \\
--dynet-mem 2048;
""", file=script_f)



if __name__ == '__main__':
    main()

