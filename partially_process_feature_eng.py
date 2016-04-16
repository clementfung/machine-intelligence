import sys
import os
import pandas as pd

from feature_eng import FeatureFactory, FeatureGenerator

if __name__ == '__main__':
    """
    To use:
    ```
    python partially_process_feature_eng.py <Class_name_to_process>,<more_class> <input_file> <output_file> <preprocess or some other string>
    ```
    """
    subs = FeatureGenerator.__subclasses__()
    ignore_list = []
    feature_classes = sys.argv[1].split(',')
    input_file  = sys.argv[2]
    output_file = sys.argv[3]
    preprocess = sys.argv[4] == 'preprocess'
    if not 'ALL' in feature_classes:
        # force the script to process all

        for obj in FeatureFactory(set_params=False).feature_generators:
            # the class you want to process
            if not obj.__class__.__name__ in feature_classes:
                ignore_list.append(obj.__class__)
    ff = FeatureFactory(
            ignore_features=ignore_list, 
            corpus_csv='data/clean_product_descriptions.csv', 
            pickle_path='pickles/'
            )
    print ff.get_feature_names()
    df_in = pd.read_csv(input_file, encoding='ISO-8859-1')
    if preprocess == True:
        df_in = ff.preprocess_columns(df_in, verbose=True)
    df_out = ff.apply_feature_eng(df_in, verbose=True)
    new_cols = ff.get_feature_names()
    if os.path.isfile(output_file):
        df_prev = pd.read_csv(output_file)
        df_prev[new_cols] = df_out[new_cols]
        df_prev.to_csv(output_file)
    else:
        keep_cols = ['id']
        if 'relevance' in df_out.columns.tolist():
            keep_cols.append('relevance')
        df_out[keep_cols + new_cols].to_csv(output_file)

    import pdb; pdb.set_trace()



