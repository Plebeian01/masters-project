from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import sys
import os

# Adjust sys.path to include the project root (one level up from 'app' directory)
# This allows importing from the 'models' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Attempting to load recommendation model components from 'models' module...")
try:
    from models import recommendationmodel as rec_model

    # Access the required components from the imported module's namespace
    xgb_model_global = rec_model.xgb_model
    movie_features_df_global = rec_model.movie_features_df
    user_factors_df_global = rec_model.user_factors_df
    # X_train is used to get column names. Ensure it's available.
    # If X_train itself is too large and only columns are needed,
    # it would be better if recommendationmodel.py stored X_train.columns separately.
    # For now, let's assume X_train is available and its columns can be accessed.
    # If X_train is not in rec_model's namespace directly after import, this will fail.
    # This was X_train.columns in the original script.
    # The script stores X.columns before train_test_split, let's use that.
    original_X_columns_global = rec_model.X.columns

    title_to_movie_id_global = rec_model.title_to_movie_id
    movie_id_to_title_global = rec_model.movie_id_to_title
    ratings_df_global = rec_model.ratings_df_orig # Using original ratings for broader user search
    one_hot_genre_columns_global = rec_model.one_hot_genre_columns

    # Get the list of all movie titles for the datalist in index.html
    # movies_df_orig is loaded in recommendationmodel.py
    all_movie_titles_global = rec_model.movies_df_orig['title'].unique().tolist()

    print("Successfully loaded recommendation model components.")
    MODEL_LOADED_SUCCESSFULLY = True
except ImportError as e:
    print(f"Error importing recommendationmodel: {e}")
    MODEL_LOADED_SUCCESSFULLY = False
except AttributeError as e:
    print(f"Error accessing attribute from recommendationmodel: {e}")
    print("This might mean some variables were not defined globally in recommendationmodel.py or not available after import.")
    MODEL_LOADED_SUCCESSFULLY = False
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    MODEL_LOADED_SUCCESSFULLY = False


app = Flask(__name__)

@app.route('/')
def index():
    if not MODEL_LOADED_SUCCESSFULLY:
        return "Error: Recommendation model components could not be loaded. Please check the server logs.", 500
    return render_template('index.html', movie_titles=all_movie_titles_global)

@app.route('/recommend', methods=['POST'])
def recommend():
    if not MODEL_LOADED_SUCCESSFULLY:
        return "Error: Recommendation model components could not be loaded. Please check the server logs.", 500

    new_user_ratings_input = []
    i = 1
    while True:
        movie_title_key = f'movie_title_{i}'
        movie_rating_key = f'movie_rating_{i}'
        if movie_title_key in request.form and movie_rating_key in request.form:
            title = request.form[movie_title_key]
            try:
                rating = float(request.form[movie_rating_key])
                if 0.5 <= rating <= 5.0:
                    new_user_ratings_input.append((title, rating))
                else:
                    # Handle invalid rating range if necessary, or rely on HTML form validation
                    pass
            except ValueError:
                # Handle non-numeric rating if necessary
                pass
            i += 1
        else:
            break

    if not new_user_ratings_input:
        # Handle case with no valid inputs, maybe redirect back with a message
        return render_template('recommendations.html', recommendations=None)

    print(f"Received user input: {new_user_ratings_input}")

    try:
        recommendations = rec_model.get_new_user_recommendations(
            new_user_ratings_input=new_user_ratings_input,
            n=5, # Number of recommendations
            model=xgb_model_global,
            all_movie_features_df=movie_features_df_global,
            historical_user_factors_df=user_factors_df_global,
            original_X_columns=original_X_columns_global,
            title_to_movie_id_map=title_to_movie_id_global,
            movie_id_to_title_map=movie_id_to_title_global,
            ratings_df=ratings_df_global, # This is ratings_df_orig in our global load
            genre_columns=one_hot_genre_columns_global
            # min_ratings_threshold, abs_pred_rating_threshold, etc., will use defaults in the function
        )
        print(f"Generated recommendations: {recommendations}")
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        recommendations = None

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    # Check if model loading failed and inform the user if running directly
    if not MODEL_LOADED_SUCCESSFULLY:
        print("FATAL: Recommendation model could not be loaded. The application cannot start properly.")
        print("Please check errors related to 'recommendationmodel.py' import and file paths.")
        # Optionally, exit if running in a script mode where this is critical
        # sys.exit(1)

    # Make sure the dataset path is correct.
    # The recommendationmodel.py uses '../dataset/100kDataset/movies.csv'
    # If app.py is in project_root/app.py, then dataset should be project_root/../dataset which is WRONG.
    # If app.py and recommendationmodel.py are in project_root/, then ../dataset means one level above project_root.
    # This needs to be resolved. For now, assuming it finds it or paths are adjusted in recommendationmodel.py.
    # A common pattern is to have a 'data' or 'datasets' folder within the project root.
    # e.g. project_root/datasets/100kDataset/*
    # And then recommendationmodel.py would use './datasets/100kDataset/' or similar.

    # The path issue is critical. If `recommendationmodel.py` cannot find its data files,
    # the import `import recommendationmodel as rec_model` will fail during Flask's initialization.

    # For development, it's often easier to run the Flask app from the root of the project.
    # If recommendationmodel.py is also in the root, and it uses '../dataset',
    # this implies the 'dataset' folder is a sibling to the project root.
    # Example:
    # parent_directory/
    #   project_root/
    #     app.py
    #     recommendationmodel.py
    #     templates/
    #     static/
    #   dataset/
    #     100kDataset/

    # This structure is a bit unusual. Usually, data is inside the project.
    # If the dataset is inside the project, e.g., project_root/dataset/100kDataset
    # Then recommendationmodel.py should use "dataset/100kDataset" (relative to itself)

    app.run(debug=True)
