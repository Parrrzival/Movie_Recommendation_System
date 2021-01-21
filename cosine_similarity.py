from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London","Paris Paris London"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

print("Count Matrix:\n",count_matrix,end = '\n')
print("Count Matrix Array : \n",count_matrix.toarray(),end='\n')

similarity_scores = cosine_similarity(count_matrix)

print("Similarity Score :\n",similarity_scores)