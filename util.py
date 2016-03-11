import pandas as pd


def join_raw(df, 
        desc_path = '',
        attr_path = '',
        ):
    df_attr = pd.read_csv(attr_path)
    df_desc = pd.read_csv(desc_path)

    og_cols = df.columns.values.tolist()
    description_cols = df_desc.columns.values.tolist()

    df = df.join(df_attr, on = "product_uid", rsuffix='_attr')
    df = df.join(df_desc, on = "product_uid", rsuffix='_desc')
    
    df_new = pd.DataFrame() 

    for g, df_g in df.groupby("product_uid"):
        print "Grouping attributes for product uid:", g
        names = df_g["name"].tolist()
        values = df_g["value"].tolist()
        attributes = [(names[i], values[i]) for i in xrange(len(df_g))]
        df_row = df_g[og_cols + description_cols]
        df_row["attributes"] = str(attributes)
        df_new = pd.concat([df_new, df_row])

    return df_new


