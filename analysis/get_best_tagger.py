import sys
import os
import shutil
import glob
import pandas

def main():
    fn = sys.argv[1]
    best = pandas.read_csv(fn)

    print(best.columns)
    script_f = open("run_test_tag.sh", "w")

    for _, row in best.iterrows():
        print(row['strategy'])
        print(row['best_valid_f1'])

        files = sorted(list(glob.glob(f"{row['fn_prefix']}*{row['Run ID']}*")))

        if len(files):
            in_f = files[-1]
            print(in_f)
            out_f = sys.argv[2] + os.path.basename(in_f)
            shutil.copy(in_f, out_f)
            
            print(f"""
echo "{row['strategy']}\t{out_f}";
build/tagger --test \\
--saved-model {out_f} \\
--dataset {row['dataset']} \\
--dim {row['dim']} \\
--batch-size {row['batch_size']} \\
--gcn-layers {row['gcn_layers']} \\
--drop 0 \\
--tree {row['strategy']} \\
--budget {row['gcn_budget']} \\
--sparsemap-eta {row['SM_eta']} \\
--sparsemap-max-iter {row['SM_maxit']} \\
--sparsemap-active-set-iter {row['SM_ASET_maxit']} \\
--sparsemap-residual-thr 1e-4 \\
--dynet-mem 1024;
""", file=script_f)


if __name__ == '__main__':
    main()

