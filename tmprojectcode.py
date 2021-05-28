import pandas as pd
import numpy as np
import os
import time
from datetime import datetime as DateTime
import transformers
from bert_score import BERTScorer
# from multiprocessing import Pool
from pandas_multiprocess import multi_process
# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

np.random.seed(123)
# required to be able to download the language models from huggingface when Zscaler security is on
# os.environ['REQUESTS_CA_BUNDLE'] = "C:\\Users\\Advait Godbole\\.pythoncacert\\cacert.pem"

## uncomment for reading full data
PATH = os.getcwd()

DATA_PATH = os.path.join(PATH, 'testset\\')
# print(DATA_PATH)
data_dir_list = os.listdir(DATA_PATH)
data_dir_list.remove('.DS_Store')
data_dir_list.remove('.ipynb_checkpoints')
# data_dir_list

datadict = {}
sampled_data_dict={}

for folder in data_dir_list:
    datadict[folder] = pd.read_csv(DATA_PATH+folder+'\scores.csv')
    # print(datadict[folder].head())

def delete_files_in_dir(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))
# exit()
def sample_df(df,nrows):
    return df.sample(n=nrows)

def get_partial_data(df,endrow,startrow=None):
    if startrow is None:
        startrow=0

    df = df[startrow:endrow]
    # else:
    #     df = [startrow:endrow]
    
    return df
# ### data loading complete

# for langpair in datadict.keys():
#     df = get_partial_data(
#         datadict[langpair],
#         endrow=100
#     )
    # print(df)

# exit()
def func(data_row,bertlang,scorer,metricname,dfpart):
    # data_row (pd.Series): a row of a panda Dataframe
    # print(f"\nInside the parallel scoring function..Applying func to part {dfpart+1}..")

    def my_bert_scorer(
        hypothesis, reference,scorer=scorer
    ):
        P,R,F1 = scorer.score(
            [hypothesis],[reference]
        )

        def convert_0_dim_tensor_to_num(zero_dim_tensor):
            return(
                zero_dim_tensor[0].item()
            )

        return(
            convert_0_dim_tensor_to_num(P),
            convert_0_dim_tensor_to_num(R),
            convert_0_dim_tensor_to_num(F1)
        )
    
    P,R,F1 = my_bert_scorer(
        data_row['translation'],
        data_row['reference']
    )

    # data_row['bert_metric_p']=P
    # data_row['bert_metric_r']=R
    # data_row['bert_metric_f']=F1
    data_row[metricname]=P
    data_row[metricname]=R
    data_row[metricname]=F1
    
    return data_row

def score_lang(
    datadict,
    # dfsize,
    langpair,
    dfparts=10
):
    # if dfsize>0:
    #     df = datadict[langpair][0:dfsize]
    # else:
    #     df = datadict[langpair]
    try:
        lenDf = len(datadict[langpair])
    except TypeError:
        print(f"\n---- len(datadict[langpair]) cannot be calculated")
        exit()
    
    print(f"\n--- Scoring for {langpair} on {lenDf} records")
    destinationLang = langpair.split("-")[1]
    
    start = time.time()
    scorer = BERTScorer(lang=destinationLang, rescale_with_baseline=True)
    # The `args` will be passed to the additional arguments of `func()`
    datadict[langpair] = datadict[langpair].replace(np.nan, '', regex=True)
    
    # exit()
    df_split = np.array_split(
        datadict[langpair],
        dfparts
    )
    # print(datadict[langpair])
    i=0
    for splitdf in df_split:
        try:
            lenSplitDf = len(splitdf)
        except TypeError:
            print(f"\n---- len(splitdf) cannot be calculated")
            exit()

        print(f"\nProcessing part {i+1} of {dfparts} containing {lenSplitDf} records...")

        args = {
            # 'bertlang':'en'#,
            'scorer':scorer,
            'bertlang':destinationLang,
            'metricname':'metric_scores',
            'dfpart':i
        }
        try:
            splitdf = multi_process(
                func=func,
                data=splitdf,
                num_process=4,
                **args
            )
            print(f"\nSaving part {i+1} to temp disk..")
        except TypeError:
            print(f"\n***** part {i+1} failed ***")
        # print(splitdf)
        splitdf.to_pickle(f"./temp/splitdf_{langpair}_{i}.pkl")
        i = i+1

    # df_split = list(
    #     map(
    #         lambda splitdf: multi_process(
    #             func=func,
    #             data=splitdf,
    #             num_process=4,
    #             **args
    #         ),
    #         df_split
    #     )
    # )
    # print(df_split)
    # result = pd.concat(
    #     [
    #         multi_process(
    #             func=func,
    #             data=splitdf,
    #             num_process=4,
    #             **args
    #         ) 
    #         for splitdf in df_split 
    #     ]
    # )
    # result = pd.concat(df_split)
    # print(result)
    # exit()
    # result = multi_process(
    #     func=func,
    #     data=datadict[langpair],
    #     num_process=4,
    #     **args
    # )
    del df_split

    TEMP_PATH = os.path.join(PATH, 'temp\\')
    result = pd.concat(
        [
            pd.read_pickle(
                TEMP_PATH+f"splitdf_{langpair}_{i}.pkl"
            ) 
            for i in range(dfparts)
        ]
    )
    
    # delete_files_in_dir(TEMP_PATH)
    # print(df_split)
    end = time.time()
    # exit()
    print("Time elapsed:", end - start)
    # print(result)
    # result.to_pickle(f"./results_test_set/result_bertscore_{langpair}testset.pkl")
    print(f"\nCompleted scoring {langpair}...Writing results..")
    result.to_csv(f"./results_test_set/result_bertscore_{langpair}_testset.csv",index=False)
    print(f"\nResults written for {langpair}..")

def main(
    startrow,
    endrow,
    skiplang=None,
    sample_data=True,
    # dfsize
    sample_rows=2,
    dfparts=10
):
    # df = pd.read_pickle("./df_10row.pkl")
    if sample_data:
        for k in datadict.keys():
            datadict[k] = sample_df(
                datadict[k],
                nrows=sample_rows
            )
    else:
        pass
    
    if endrow is not None:
        for k in datadict.keys():
            datadict[k] = get_partial_data(
                datadict[k],
                endrow=endrow,
                startrow=startrow
            )
            # print(datadict[k])

    # exit()
    
    startfull = time.time()
    print(f"\nStarted the process at {DateTime.now()}")
    if skiplang is None:
        validItems = list(datadict.keys())
    else: 
        fullLangpairList = list(datadict.keys())
        validItems = [
            langpairItem for langpairItem in fullLangpairList 
            if langpairItem not in skiplang
        ]
    try:
        lenValidItems = len(validItems)
    except TypeError:
        print(f"\n---- len(validItems) cannot be calculated")
        exit()

    print(f"\nCalculating scores for {lenValidItems} language pairs: {validItems}")

    for langpair in validItems:
        score_lang(
            datadict,
            # dfsize,
            langpair,
            dfparts
        )
    
    endfull = time.time()
    try:
        lenDataDict = len(datadict)
    except TypeError:
        print(f"\n---- len(datadict) cannot be calculated")
        exit()

    print(f"Total Elapsed Time for {lenDataDict} DF elapsed: {endfull - startfull}")

if __name__ == '__main__':
    # main(dfsize=100)
    main(
        sample_data=False,
        sample_rows=10,
        startrow=None,
        endrow=None,
        dfparts=20#,
        # skiplang=[
        #     'cs-en',
        #     'de-en',
        #     'en-fi',
        #     'en-zh',
        #     'ru-en',
        #     'zh-en'#,
        # ]
    )