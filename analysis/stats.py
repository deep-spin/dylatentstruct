import sys
import pandas


def nli():
    fn = sys.argv[1]
    data = pandas.read_csv(fn)
    cols = ["fn_prefix", "Run ID", "Name", "attention", "lr", "best_valid_acc"]

    test_cols = ["fn_prefix", "Run ID", "attention", "SM_ASET_maxit", "SM_eta", "SM_maxit", "SM_thr",
    "batch_size", "dataset", "normalize_embed", "update_embed",
    "best_valid_acc"]

    data = data.ix[('reimp' in fn for fn in data.fn_prefix)]
    best_ix = []
    # cols = ['Name', 'SM_eta', 'SM_maxit', 'lr', 'strategy', 'gcn_budget', 'gcn_layers', 'valid_f1']
    #print(data[cols].groupby("attention"))

    gb = data.groupby("attention")
    for attn, df in gb:
        print(df[cols].sort_values("lr"))
        ix = df["best_valid_acc"].idxmax()
        best_ix.append(ix)


    best = data[test_cols].loc[best_ix]
    print(best)
    best.to_csv(f"best-{fn}")


def tag():
    fn = sys.argv[1]
    data = pandas.read_csv(fn)
    cols = ["fn_prefix", "Run ID", "Name", "strategy", "lr", "best_valid_f1",
    "gcn_budget", "SM_eta", "dim"]

    test_cols = ["fn_prefix", "Run ID", "Name", "strategy", "SM_ASET_maxit", "SM_eta", "SM_maxit", "SM_thr",
    "batch_size", "dataset", "best_valid_f1", "dim"]

    # data = data.ix[('reimp' in fn for fn in data.fn_prefix)]
    best_ix = []
    # cols = ['Name', 'SM_eta', 'SM_maxit', 'lr', 'strategy', 'gcn_budget', 'gcn_layers', 'valid_f1']
    #print(data[cols].groupby("attention"))

    gb = data.groupby("Name")
    for attn, df in gb:
        print(df[cols].sort_values("lr"))
        ix = df["best_valid_f1"].idxmax()
        best_ix.append(ix)


    best = data[test_cols].loc[best_ix]
    print(best)
    best.to_csv(f"best-{fn}")


if __name__ == '__main__':
    nli()
    # tag()

