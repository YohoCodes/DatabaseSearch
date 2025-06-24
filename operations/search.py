from sentence_transformers import SentenceTransformer, util
import numpy as np

class Search:
    def __init__(self, database, search_by: str, search_res: str | list[str], res_is_by = True):
        self.model          = SentenceTransformer("all-MiniLM-L6-v2")
        self.database       = database # a dataframe like object
        self.by_arr         = database[search_by].values # 1-d array
        self.res_is_by      = res_is_by # is the result column equal to search by column

        self.search_res     = search_res 
        self.res_is_str     = isinstance(self.search_res, str) # boolean checks result type
        
        if res_is_by: # short for 'result is by', meaning the result column and by column are the same
        # If true, search returns data from by column
            self.res_arr    = self.by_arr
        else:           
        # If false, search returns data from selected columns but still based by column embeddings
            self.res_arr    = database[search_res].values

        # calculates database vectors, group
        self.by_embeddings  = self.model.encode(self.by_arr) # Tuple_i structure: (by, by_embedding, res)
        

    def top_results(self, query: str, n_results = 5):

        # calculate the query vector
        query_embedding     = self.model.encode(query) 

        # list of cosine similarities between query embedding and each search_by column value embedding
        similarities        = util.cos_sim(query_embedding, self.by_embeddings)[0].numpy()

        # Located indexes of results with highest cosine similarity according to search_by column
        top_idx             = np.argsort(similarities)[::-1][:n_results]

        
        if self.res_is_str:
            # Return top results that correspond with these indeces
            top_res = self.res_arr[top_idx]

        else:
            # Return a list of arrays yielding top results based on search_by
            top_res = [arr[top_idx] for arr in self.res_arr.T]
                
        if self.res_is_by:
            # If search_by == search_res, return top result only
            return top_res
        
        else:
            # Ensure both arrays are 2D for concatenation
            by_arr_2d = np.expand_dims(self.by_arr[top_idx], axis=0)
            arrays = np.concatenate((by_arr_2d, top_res), axis=0)

            return zip(*arrays)  # Unpack arrays into tuples for each result


if __name__ == '__main__':
    import pandas as pd
    
    print('Test block executed')
    df = pd.read_csv('../SethGodin.csv').dropna()

    by_column = 'title'

    # To test: Can res_column take list of strings?
    res_column = ['url', 'publication-date']

    if by_column == res_column:
        by_equal_res = True
    else:
        by_equal_res = False

    title_search = Search(df, search_by=by_column, search_res=res_column, res_is_by=by_equal_res)

    print(list(title_search.top_results('Work is hard')))