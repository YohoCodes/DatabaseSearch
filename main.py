from operations import Search

import pandas as pd
import numpy as np
import streamlit as st

'''

This is a database search engine. You can search through large databases and
return column values that most closely resemble your query.

'''

# When search_engine_type = 'Amazon', fill by_column and res_column using these values
amazon_column_names = ['PRODUCT_ID', 'TITLE', 'BULLET_POINTS', 'DESCRIPTION',
'PRODUCT_TYPE_ID', 'PRODUCT_LENGTH']

# When search_engine_type = 'SethGodin', fill by_column and res_column using these values
seth_godin_column_names = ['id', 'url', 'title', 'content_plain', 'content_html',
'stars', 'publication-date', 'referral-url']

# Unless you are cleaning data, leave as is...
stage = 'create_webpage'

######## CONFIGURE SEARCH ENGINE BELOW ########

# Must match name of .csv file in root directory
search_engine_type = 'Amazon'  # 'Amazon' or 'SethGodin'

# Query will be compared to the by_column values
# This column will always be returned in search results
by_column = 'TITLE'

# Configure what results to show
# Accepts str or list[str]
# Set equal to by_column variable if you don't need to display additional result columns
res_column = ['BULLET_POINTS', 'DESCRIPTION', 'PRODUCT_LENGTH']

# Configure how many results to show
number_of_results = 5

######## CONFIGURE SEARCH ENGINE ABOVE ########

if by_column == res_column:
    by_equal_res = True
else:
    by_equal_res = False


# Using this area to create webpage
if stage == 'create_webpage':

    @st.cache_resource
    def get_engine():
        """
        Caches the Search engine instance using Streamlit's @st.cache_resource decorator
        to avoid re-initializing it on every rerun, which improves performance when the data does not change.

        Caveat: If the underlying data changes, the cached Search engine will not reflect those changes
        unless the cache is manually cleared or the app is restarted.
        """

        data = pd.read_csv(f'{search_engine_type}.csv')

        return Search(data, search_by=by_column, search_res=res_column, res_is_by=by_equal_res)
    
    with st.spinner('Initializing search engine...'):
        engine = get_engine()

    st.title(f"{search_engine_type} Search Engine")
    
    query = st.text_input("Enter your search query:", "")

    if query:
        results = engine.top_results(query, n_results=number_of_results)

        for i, res in enumerate(results):

            if by_equal_res:
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <strong>Result&nbsp;{i+1}</strong>
                    </div>

                    {res_column}:<br>{res}
                    """,
                    unsafe_allow_html=True,
                )

            else:
                if engine.res_is_str:
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <strong>Result&nbsp;{i+1}</strong>
                        </div>

                        {by_column}:<br>{res[0]}<br><br>
                        {res_column}:<br>{res[1]}
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # res[0]  -> single value from the search-by column
                    # res[1]  -> list of values, one per column in res_column
                    per_column_html = "".join(
                        f"{col}:<br>{val}<br><br>"
                        for col, val in zip(res_column, res[1:])
                    )

                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <strong>Result&nbsp;{i+1}</strong>
                        </div>

                        {by_column}:<br>{res[0]}<br><br>
                        {per_column_html}
                        """,
                        unsafe_allow_html=True,
                    )
                    


    else:
        st.write('Please enter a query')