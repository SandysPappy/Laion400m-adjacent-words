import pandas as pd
from pandas import read_parquet
import argparse
from laion400m_test_strings import get_test_strings
import regex as re
import string
import numpy as np
from tqdm import tqdm
import gc


def remove_characters_np(text_array, chars_to_remove):
    for char in chars_to_remove:
        text_array = np.char.replace(text_array, char, "")
    return text_array




def get_word_inds() -> list[str]:
    strs = get_words_toy()
    reg_wht_spc = "\s"

    # remove all punctuation except \ - _ and /
    fltr = "!\"#$%&\'()*+,.:;<=>?@[]^`{|}~"

    for s_idx, s in enumerate(strs):
        substrs = re.split(reg_wht_spc, s)
        for s_sub_idx, s_sub in enumerate(substrs):
            pass

# done first
def build_no_white_space_col(df: pd.DataFrame, batch_size=10000) -> pd.DataFrame:
    n, _ = df.shape # rows, cols

        # Iterate through the DataFrame in batches
    for start in tqdm(range(0, n, batch_size), desc="Building TEXT_NO_WHITE_SPACE column"):
        end = min(start + batch_size, n)
        # Apply transformation to the selected batch
        df.loc[start:end, "TEXT_NO_WHITE_SPACE"] = df.iloc[start:end, 2].apply(lambda x: re.split(r'\s', x) if isinstance(x, str) else [])

    return df

# # ignore the -_/\ characters
# def remove_most_punctuation(arr: np.array) -> np.array:
#     fltr = "!\"#$%&\'()*+,.:;<=>?@[]^`{|}~"
#     return np.array([
#         s.translate(str.maketrans('', '', fltr)) if isinstance(s, str) else ""
#         for s in arr
#     ])

def build_no_punctuation(df: pd.DataFrame, batch_size=10000) -> pd.DataFrame:
    n, _ = df.shape # rows, cols

    # Iterate through the DataFrame in batches
    for start in tqdm(range(0, n, batch_size), desc="Building TEXT_NO_PUNC column"):
        end = min(start + batch_size, n)
        
        # ignore the -_/\ characters
        fltr = "!\"#$%&\'()*+,.:;<=>?@[]^`{|}~"
        df.loc[start:end, "TEXT_NO_PUNC"] = df.loc[start:end, "TEXT_NO_WHITE_SPACE"].apply(lambda lst: [s.translate(str.maketrans('', '', fltr)) if isinstance(s, str) else "" for s in lst])

    return df


# paths need to be same length
# writes output parquet files to use later
def search_pipeline(parquet_paths: list[str], out_paths: list[str]) -> None:

    for idx, f in tqdm(enumerate(parquet_paths), desc="Search Pipeline Time"):
        df = read_parquet(f)
        # df = build_no_white_space_col(df, batch_size=10000)
        df = build_no_punctuation(df=df, batch_size=10000)

        df.to_parquet(out_paths[idx])
        del df
        gc.collect()

def get_words_toy() -> list[str]:

    example_small = [
        "HALO 3: ODST  (XBOX 360, 2009)(9741) **FREE SHIPPING USA*SHIPS NEXT BUSINESS DAY",
        "The Queen of the Damned (Vampire Chronicles Series #3) ",
        "Modern C++ Design: Generic Programming and Design Patterns Applied",
        "Death-Race-v3 icon",
        "2009 Jeep Patriot EV. Image by Jeep.",
        "Harvey Horrors Collected Works WITCHES TALES VOL #2 TPB Softie Comics #6-10 TP",
        "The feeling of belonging Do you create opportunities for staff and clients to cooperate on programs.",
        "Makita Variable Speed Polisher-Power Tools-Makita-9237C",
        "\"\"Putin on the forum of \"\"One belt and one road\"\". Video\"\"",
        "",
        " ",
        "###",
        " ] "
    ]
    return example_small


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--parq_paths",
                        nargs='+',
                        default= [
                            '../../laion400m/laion400-meta/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00001-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00002-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00003-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00004-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00005-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00006-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00007-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00008-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00009-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00010-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00011-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00012-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00013-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00014-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00015-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00016-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00017-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00018-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00019-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00020-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00021-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00022-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00023-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00024-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00025-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00026-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00027-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00028-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00029-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00030-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                            '../../laion400m/laion400-meta/part-00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
                        ],
                        help="Paths to input Parquet files."
    )
    
    parser.add_argument("--out_paths",
                        nargs='+',
                        default= [
                            "search_pipeline_out_0.parquet",
                            "search_pipeline_out_1.parquet",
                            "search_pipeline_out_2.parquet",
                            "search_pipeline_out_3.parquet",
                            "search_pipeline_out_4.parquet",
                            "search_pipeline_out_5.parquet",
                            "search_pipeline_out_6.parquet",
                            "search_pipeline_out_7.parquet",
                            "search_pipeline_out_8.parquet",
                            "search_pipeline_out_9.parquet",
                            "search_pipeline_out_10.parquet",
                            "search_pipeline_out_11.parquet",
                            "search_pipeline_out_12.parquet",
                            "search_pipeline_out_13.parquet",
                            "search_pipeline_out_14.parquet",
                            "search_pipeline_out_15.parquet",
                            "search_pipeline_out_16.parquet",
                            "search_pipeline_out_17.parquet",
                            "search_pipeline_out_18.parquet",
                            "search_pipeline_out_19.parquet",
                            "search_pipeline_out_20.parquet",
                            "search_pipeline_out_21.parquet",
                            "search_pipeline_out_22.parquet",
                            "search_pipeline_out_23.parquet",
                            "search_pipeline_out_24.parquet",
                            "search_pipeline_out_25.parquet",
                            "search_pipeline_out_26.parquet",
                            "search_pipeline_out_27.parquet",
                            "search_pipeline_out_28.parquet",
                            "search_pipeline_out_29.parquet",
                            "search_pipeline_out_30.parquet",
                            "search_pipeline_out_31.parquet",
                        ],
                        help="Paths to output Parquet files."
    )

    args = parser.parse_args()

    parq_paths: list[str] = args.parq_paths
    out_paths: list[str] = args.out_paths

    # swapping for the removing punctuation 
    parq_paths = [
                            "no_white_spaces/search_pipeline_out_0.parquet",
                            "no_white_spaces/search_pipeline_out_1.parquet",
                            "no_white_spaces/search_pipeline_out_2.parquet",
                            "no_white_spaces/search_pipeline_out_3.parquet",
                            "no_white_spaces/search_pipeline_out_4.parquet",
                            "no_white_spaces/search_pipeline_out_5.parquet",
                            "no_white_spaces/search_pipeline_out_6.parquet",
                            "no_white_spaces/search_pipeline_out_7.parquet",
                            "no_white_spaces/search_pipeline_out_8.parquet",
                            "no_white_spaces/search_pipeline_out_9.parquet",
                            "no_white_spaces/search_pipeline_out_10.parquet",
                            "no_white_spaces/search_pipeline_out_11.parquet",
                            "no_white_spaces/search_pipeline_out_12.parquet",
                            "no_white_spaces/search_pipeline_out_13.parquet",
                            "no_white_spaces/search_pipeline_out_14.parquet",
                            "no_white_spaces/search_pipeline_out_15.parquet",
                            "no_white_spaces/search_pipeline_out_16.parquet",
                            "no_white_spaces/search_pipeline_out_17.parquet",
                            "no_white_spaces/search_pipeline_out_18.parquet",
                            "no_white_spaces/search_pipeline_out_19.parquet",
                            "no_white_spaces/search_pipeline_out_20.parquet",
                            "no_white_spaces/search_pipeline_out_21.parquet",
                            "no_white_spaces/search_pipeline_out_22.parquet",
                            "no_white_spaces/search_pipeline_out_23.parquet",
                            "no_white_spaces/search_pipeline_out_24.parquet",
                            "no_white_spaces/search_pipeline_out_25.parquet",
                            "no_white_spaces/search_pipeline_out_26.parquet",
                            "no_white_spaces/search_pipeline_out_27.parquet",
                            "no_white_spaces/search_pipeline_out_28.parquet",
                            "no_white_spaces/search_pipeline_out_29.parquet",
                            "no_white_spaces/search_pipeline_out_30.parquet",
                            "no_white_spaces/search_pipeline_out_31.parquet",
                        ]
    out_paths = [
                            "no_punc/search_pipeline_out_nopunc_0.parquet",
                            "no_punc/search_pipeline_out_nopunc_1.parquet",
                            "no_punc/search_pipeline_out_nopunc_2.parquet",
                            "no_punc/search_pipeline_out_nopunc_3.parquet",
                            "no_punc/search_pipeline_out_nopunc_4.parquet",
                            "no_punc/search_pipeline_out_nopunc_5.parquet",
                            "no_punc/search_pipeline_out_nopunc_6.parquet",
                            "no_punc/search_pipeline_out_nopunc_7.parquet",
                            "no_punc/search_pipeline_out_nopunc_8.parquet",
                            "no_punc/search_pipeline_out_nopunc_9.parquet",
                            "no_punc/search_pipeline_out_nopunc_10.parquet",
                            "no_punc/search_pipeline_out_nopunc_11.parquet",
                            "no_punc/search_pipeline_out_nopunc_12.parquet",
                            "no_punc/search_pipeline_out_nopunc_13.parquet",
                            "no_punc/search_pipeline_out_nopunc_14.parquet",
                            "no_punc/search_pipeline_out_nopunc_15.parquet",
                            "no_punc/search_pipeline_out_nopunc_16.parquet",
                            "no_punc/search_pipeline_out_nopunc_17.parquet",
                            "no_punc/search_pipeline_out_nopunc_18.parquet",
                            "no_punc/search_pipeline_out_nopunc_19.parquet",
                            "no_punc/search_pipeline_out_nopunc_20.parquet",
                            "no_punc/search_pipeline_out_nopunc_21.parquet",
                            "no_punc/search_pipeline_out_nopunc_22.parquet",
                            "no_punc/search_pipeline_out_nopunc_23.parquet",
                            "no_punc/search_pipeline_out_nopunc_24.parquet",
                            "no_punc/search_pipeline_out_nopunc_25.parquet",
                            "no_punc/search_pipeline_out_nopunc_26.parquet",
                            "no_punc/search_pipeline_out_nopunc_27.parquet",
                            "no_punc/search_pipeline_out_nopunc_28.parquet",
                            "no_punc/search_pipeline_out_nopunc_29.parquet",
                            "no_punc/search_pipeline_out_nopunc_30.parquet",
                            "no_punc/search_pipeline_out_nopunc_31.parquet",
                        ]
    

    search_pipeline(parquet_paths=parq_paths, out_paths=out_paths)