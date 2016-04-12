import pandas as pd


def join_raw(df, 
        desc_path = '',
        attr_path = '',
        ):
    df_attr = pd.read_csv(attr_path)
    df_desc = pd.read_csv(desc_path)

    og_cols = df.columns.values.tolist()
    description_cols = df_desc.columns.values.tolist()

    df = pd.merge(df, df_attr, how='left', on = "product_uid")
    df = df.join(df_desc, on = "product_uid", rsuffix='_desc')
    
    df_new = pd.DataFrame() 
    rows = []

    for g, df_g in df.groupby(["product_uid", "search_term"]):
        print "Grouping attributes for product uid:", g
        names = df_g["name"].tolist()
        values = df_g["value"].fillna('').tolist()
        attributes = [(names[i], values[i]) \
                for i in xrange(len(df_g))\
                if not pd.isnull(names[i])
                ]
        # just take the first row
        df_row = df_g[og_cols + description_cols].head(n=1)
        df_row["attributes"] = str(attributes)
        df_new = pd.concat([df_new, df_row])
    return df_new.fillna('')

def flatten_to_list(a):
    """
    Takes a list of lists (mixed types) 
    to return a list of single objects
    """
    return reduce(
            lambda l,r: l + r, 
            map(lambda x: [x] if not type(x) == list else x, a)
            )
