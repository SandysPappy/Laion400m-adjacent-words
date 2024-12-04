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
    
    # init verticies for classes we care about
    vertex_cls_strs = ['man', 'men', 'male', 'women', 'woman', 'female']
    for cls_str in vertex_cls_strs:
        key_map[cls_str] = str_id
        str_id += 1
        g.add_vertex(name=str_id)
    vertex_cls_ids = [key_map[s] for s in vertex_cls_strs]

    for parq_file in tqdm(parquet_paths, desc="Processing laion400M parquet batches"):
        df = read_parquet(parq_file)
        filtered_strs = df.loc[0:200000, "TEXT_NO_PUNC"]

        # treat each contigious string in the list as a vertex
        for row in tqdm(filtered_strs, desc='Building graph on this batch'):
            
            # if len(row) > 15:
                # continue

            row = [s.lower() for s in row]

            found = any(word in row for word in vertex_cls_strs)
            if not found:
                continue

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

                # we only care about neighbors of the class string verticies. should speed this up a LOT
                if curr_vertex_id not in vertex_cls_ids:
                    continue

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

# returns a list of neighbors to this vertex string ordered by weight
def get_neighbors(g: Graph, key_map: dict[str, int], vertex_str: str) -> list[tuple[str, int]]:

    if vertex_str not in key_map:
        return None

    vert_id = key_map[vertex_str]
    neighbor_ids = g.neighbors(vert_id)

    all_neighbors = []

    for neighbor_id in neighbor_ids:
        # Integer to find the corresponding key for
        target_int = neighbor_id
        # Find the key (vertex_str) corresponding to the target integer
        neighbor_str = next((key for key, value in key_map.items() if value == target_int), None)
        
        edge_id = g.get_eid(vert_id, neighbor_id)
        edge_weight = g.es[edge_id]["weight"]

        all_neighbors.append((neighbor_str, edge_weight))

    # order so largest weights are first
    all_neighbors = sorted(all_neighbors, key=lambda x: x[1], reverse=True)

    return all_neighbors

# return vertex ids of vertices which connect to all the man, men, and male vertices where the edge weight is also above a threshold
def filter_men_vertices(graph: Graph, key_map: dict[str, int], weight_threshold) -> list[int]:

    valid_verts = []

    num_verts = len(graph.vs)
    for i in range(num_verts):
        try:
            edge_id1 = graph.get_eid(i, 0) # "man"
            edge_id2 = graph.get_eid(i, 1) # "men"
            edge_id3 = graph.get_eid(i, 2) # "male"

            if graph.es[edge_id1]["weight"] > weight_threshold and graph.es[edge_id2]["weight"] > weight_threshold and graph.es[edge_id3]["weight"] > weight_threshold:
                valid_verts.append(i)

        # no edge
        except ValueError:
            continue

    return valid_verts

# return vertex ids of vertices which connect to all the man, men, and male vertices where the edge weight is also above a threshold
def filter_women_vertices(graph: Graph, key_map: dict[str, int], weight_threshold) -> list[int]:

    valid_verts = []

    num_verts = len(graph.vs)
    for i in range(num_verts):
        try:
            edge_id1 = graph.get_eid(i, 3) # "women"
            edge_id2 = graph.get_eid(i, 4) # "woman"
            edge_id3 = graph.get_eid(i, 5) # "female"

            if graph.es[edge_id1]["weight"] > weight_threshold and graph.es[edge_id2]["weight"] > weight_threshold and graph.es[edge_id3]["weight"] > weight_threshold:
                valid_verts.append(i)

        # no edge
        except ValueError:
            continue

    return valid_verts




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_parquet_paths",
                        nargs='+',
                        default= [
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_0.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_1.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_2.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_3.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_4.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_5.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_6.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_7.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_8.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_9.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_10.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_11.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_12.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_13.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_14.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_15.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_16.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_17.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_18.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_19.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_20.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_21.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_22.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_23.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_24.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_25.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_26.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_27.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_28.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_29.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_30.parquet",
                            "regex_handlers/no_punc/search_pipeline_out_nopunc_31.parquet",
                        ],
                        help="Paths to output Parquet files."
    )

    args = parser.parse_args()

    parq_path = args.filtered_parquet_paths

    graph, key_map = construct_graph(parquet_paths=args.filtered_parquet_paths)

    save_graph_and_keymap(graph=graph, key_map=key_map, graph_file='results/laion400M_graph_malefemale_neighbors_only_200000_ALL.pkl',  keymap_file='results/key_map_malefemale_neighbors_only_200000_ALL.pkl.gz')