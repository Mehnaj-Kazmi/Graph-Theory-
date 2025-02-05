import numpy as np
import pandas as pd
import random
import networkx as nx
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# Generate random data for 300 users and 300 movies
num_users = 300
num_movies = 300
ratings = []

# Randomly assign ratings from 1-5 to a subset of user-movie pairs
for _ in range(5000):  # Generate 5,000 interactions
    user = random.randint(1, num_users)
    movie = random.randint(1, num_movies)
    rating = random.randint(1, 5)
    ratings.append((user, movie, rating))

# Convert ratings to a DataFrame
ratings_df = pd.DataFrame(ratings, columns=["User", "Movie", "Rating"])

# Remove duplicate entries, keeping the latest rating for each User-Movie pair
ratings_df = ratings_df.drop_duplicates(subset=["User", "Movie"], keep="last")

# Scale ratings to 1-10 (multiplying by 2 for simplicity)
ratings_df["Rating"] = ratings_df["Rating"] * 2

# Create a User-Movie interaction matrix
interaction_matrix = ratings_df.pivot(index="User", columns="Movie", values="Rating").fillna(0)
interaction_array = interaction_matrix.to_numpy()

# Perform matrix factorization using NMF
nmf = NMF(n_components=15, init='random', random_state=0, max_iter=500)
user_factors = nmf.fit_transform(interaction_array)
movie_factors = nmf.components_

# Reconstruct the matrix
reconstructed_matrix = np.dot(user_factors, movie_factors)

# Assign genres to each movie
movie_genres = {f"Movie_{i+1}": random.choice(
    ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Thriller"]
) for i in range(num_movies)}

# Create a bipartite graph for users and movies
B = nx.Graph()

# Add nodes
user_nodes = [f"User_{i}" for i in range(1, num_users + 1)]
movie_nodes = [f"Movie_{j}" for j in range(1, num_movies + 1)]
B.add_nodes_from(user_nodes, bipartite=0)
B.add_nodes_from(movie_nodes, bipartite=1)

# Add edges with weights as ratings
for user, movie, rating in ratings_df.values:
    B.add_edge(f"User_{int(user)}", f"Movie_{int(movie)}", weight=rating)

# Visualize the bipartite graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(B, k=0.15, iterations=20)
nx.draw(B, pos, node_size=8, edge_color="lightblue", with_labels=False)
plt.title("User-Movie Bipartite Graph")
plt.show()

# Recommend movies for a specific user
def recommend_movies(user_id, num_recommendations=5):
    """
    Recommend top movies for a given user based on reconstructed ratings.
    """
    user_idx = user_id - 1
    user_ratings = reconstructed_matrix[user_idx]
    movie_ratings = [(f"Movie_{movie+1}", rating, movie_genres[f"Movie_{movie+1}"]) for movie, rating in enumerate(user_ratings)]
    movie_ratings_sorted = sorted(movie_ratings, key=lambda x: x[1], reverse=True)[:num_recommendations]
    return movie_ratings_sorted

# Example recommendation
example_user = 1
recommendations = recommend_movies(example_user)

print(f"Recommended movies for User_{example_user}:")
for movie, rating, genre in recommendations:
    print(f"{movie} (Genre: {genre}): Predicted rating = {rating:.2f}/10")
