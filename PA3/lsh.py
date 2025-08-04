from collections import defaultdict
from dataclasses import field, dataclass

import numpy as np

SEED = 1

@dataclass
class Bucket:
    genres: set[str] = field(default_factory=set)


class HashTable:
    def __init__(self, hash_size: int, input_dimension: int):
        self.hash_size: int = hash_size
        self.input_dimension: int = input_dimension
        self.projections: np.ndarray
        # {hash : (str)}
        self.buckets: defaultdict[str, Bucket] = defaultdict(Bucket)
        self.get_random_projection_matrix(input_dimension, hash_size)

    def get_random_projection_matrix(self, input_dimension, hash_length):
        # so the matrix is reproducible
        np.random.seed(SEED)

        scale = np.sqrt(3)
        values = np.array([1, 0, -1])
        probabilitys = [1 / 6, 2 / 3, 1 / 6]

        R = np.random.choice(values, size=(input_dimension, hash_length), p=probabilitys)
        R = scale * R
        self.projections = R

    def generate_hash(self, vector) -> str:
        projection = np.dot(vector, self.projections)
        # TODO: I dont know if this is correct -> would create hashes that
        # are strings of 1s and 0s as in the blog https://medium.com/data-science/locality-sensitive-hashing-for-music-search-f2f1940ace23
        hash_bits = (projection > 0).astype(int)
        return "".join(map(str, hash_bits))

    def set_item(self, vector: np.ndarray, label: str):
        hash_value = self.generate_hash(vector)
        self.buckets[hash_value].genres.add(label)

    def get_item(self, vector: np.ndarray) -> Bucket:
        hash_value = self.generate_hash(vector)
        return self.buckets[hash_value]


class LSH:
    def __init__(self, num_tables: int, hash_size: int, input_dimension: int):
        self.num_tables = num_tables
        self.hash_tables = []
        for i in range(num_tables):
            self.hash_tables.append(HashTable(hash_size, input_dimension))

    def set_item(self, vector: np.ndarray, label: str):
        for table in self.hash_tables:
            table.set_item(vector, label)

    def query(self, vector: np.ndarray):
        """getting back a bucket with all genres accords all tables and the hashes"""
        result_bucket = Bucket()
        result_hashes = []
        # create a bucket obj out of all buckets from all tables
        for table in self.hash_tables:
            bucket = table.get_item(vector)
            result_bucket.genres.update(bucket.genres)
            result_hashes.append(table.generate_hash(vector))

        return result_hashes, result_bucket
