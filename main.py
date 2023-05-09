import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk
from sklearn.metrics.pairwise import cosine_similarity


def get_clean_dataframe(recipes_df: pd.DataFrame):
    # drop rows with empty data
    recipes_df = recipes_df.dropna()

    # keep only recipes with 'indian' or 'greek' words in their names
    updated_df = recipes_df[recipes_df['name'].str.contains('indian|greek', case=False)]

    #drop columns that we do not need
    unneeded_columns = ['id',
                        'minutes',
                        'contributor_id', 
                        'submitted', 
                        'tags', 
                        'nutrition', 
                        'n_steps', 
                        'steps', 
                        'n_ingredients', 
                        'description'
                        ]
    
    updated_df = updated_df.drop(unneeded_columns, axis=1)

    return updated_df


def save_to_file(indian_greek_recipes_df: pd.DataFrame):

    # the 'cleaned' csv file with only indian & greek recipes/ingredients inside
    OUTPUT_FILENAME = 'indian_greek_recipes.csv'

    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)

    indian_greek_recipes_df.to_csv(OUTPUT_FILENAME, index=False)


if __name__ == '__main__':

    recipes_df = pd.read_csv('RAW_recipes.csv')

    indian_greek_recipes_df = get_clean_dataframe(recipes_df)
    
    # save_to_file(indian_greek_recipes_df)

    indian_greek_recipes_df['name'] = indian_greek_recipes_df['name'].apply(lambda x: nltk.word_tokenize(x))

    # train Word2Vec
    indian_greek_recipes_df['tokens'] = [eval(recipe) for recipe in indian_greek_recipes_df.ingredients.tolist()]




    # texts = indian_greek_recipes_df.ingredients.values.tolist()
    word2vec_model = Word2Vec(sentences=indian_greek_recipes_df['tokens'], vector_size=100, window=5, min_count=1, workers=4)


    similar_words = {search_term: [item[0] for item in word2vec_model.wv.most_similar([search_term], topn=5)]
                  for search_term in ['paneer cheese','egg','mango','bread', 'rice']}
    print(similar_words)


    print('\n')
    print(word2vec_model.wv.most_similar('feta cheese'))

    quit()






































    # Vectorize each recipe to get a single vector representation on each recipe
    indian_greek_recipes_df['vectors'] = indian_greek_recipes_df['tokens'].apply(lambda tokens: np.mean([word2vec_model.wv[token] for token in tokens], axis=0))

    # Calculate cosine similarity to find which recipes are similar
    cosine_similarities = cosine_similarity(indian_greek_recipes_df['vectors'].tolist(),
                                             indian_greek_recipes_df['vectors'].tolist())

    try:
        sims = word2vec_model.wv.most_similar('feta cheese', topn=10)
        print(sims)
    except KeyError:
        print("The word 'salt' does not appear in this model")
