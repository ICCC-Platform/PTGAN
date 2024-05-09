from .distance import cross_cosine_distances, cross_euclidean_distance
from .reranking import re_ranking

metrices_map = {
    'cosine': cross_cosine_distances,
    'l2' : cross_euclidean_distance,
    're_ranking':re_ranking
}
