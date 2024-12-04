import argparse
from igraph import Graph
from pandas import read_parquet
import gc
from tqdm import tqdm
import pickle
import gzip

# returns the graph and the dict mapping strings to vertex ids
def construct_graph(parquet_paths: list[str]) -> tuple[Graph, dict[str,int]]:

    g = Graph(directed=False)

    # igraph needs ints as ids, so we need a map of strings to ids
    key_map: dict[str, int] = {}
    str_id: int= 0

    for parq_file in tqdm(parquet_paths, desc="Processing laion400M parquet files"):
        df = read_parquet(parq_file)
        filtered_strs = df.loc[0:20000, "TEXT_NO_PUNC"]

        # treat each contigious string as a vertex
        for row in tqdm(filtered_strs, desc='Building graph on this batch'):


            # if len(row) > 15:
                # continue

            verticies_in_sentence: list[int] = []

            # populate verticies in the same sentence
            for vertex_str in row:
                if vertex_str == '':
                    continue
                vertex_str = vertex_str.lower()

                if vertex_str not in key_map:
                    key_map[vertex_str] = str_id
                    str_id += 1
                    g.add_vertex(name=str_id)

                vertex_id = key_map[vertex_str]
                verticies_in_sentence.append(vertex_id)

            # add an edge between each each word within the same sentence
            for curr_idx, curr_vertex_id in enumerate(verticies_in_sentence):
               for adj_idx, adj_vertex_id in enumerate(verticies_in_sentence):
                
                # ignore if its the same vertex.
                if curr_idx == adj_idx:
                      continue 
                
                # add the edges
                if g.are_connected(curr_vertex_id, adj_vertex_id):
                    edge_id = g.get_eid(curr_vertex_id, adj_vertex_id)
                    g.es[edge_id]["weight"]+=1
                else:
                    g.add_edge(curr_vertex_id, adj_vertex_id, weight=1)

        # these dataframes are 3GB each        
        del df
        gc.collect()

                    
    return g, key_map

def save_graph_and_keymap(graph: Graph, key_map: dict[str, int], graph_file: str, keymap_file: str):
    # Save the graph
    graph.write_pickle(graph_file)  # Use .pkl.gz for compressed output
    
    # Save the key_map using pickle with gzip compression
    with gzip.open(keymap_file, 'wb') as f:
        pickle.dump(key_map, f)

def load_graph_and_keymap(graph_file: str, keymap_file: str):
    # Load the graph
    graph = Graph.Read_Pickle(graph_file)
    
    # Load the key_map
    with gzip.open(keymap_file, 'rb') as f:
        key_map = pickle.load(f)
    
    return graph, key_map

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_parquet_paths",
                        nargs='+',
                        default= [
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_0.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_1.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_2.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_3.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_4.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_5.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_6.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_7.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_8.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_9.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_10.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_11.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_12.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_13.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_14.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_15.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_16.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_17.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_18.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_19.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_20.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_21.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_22.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_23.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_24.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_25.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_26.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_27.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_28.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_29.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_30.parquet",
                            # "regex_handlers/no_punc/search_pipeline_out_nopunc_31.parquet",
                        ],
                        help="Paths to output Parquet files."
    )

    args = parser.parse_args()

    parq_path = args.filtered_parquet_paths

    graph, key_map = construct_graph(parquet_paths=args.filtered_parquet_paths)

    save_graph_and_keymap(graph=graph, key_map=key_map, graph_file='laion400M_graph.pkl',  keymap_file='key_map.pkl.gz')