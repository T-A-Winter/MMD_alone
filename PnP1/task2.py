import os, sys
from urllib.parse import urlparse
import requests

# Settings
tmp_dir = 'tmp'
urls = ['https://www.gutenberg.org/files/1661/1661-0.txt', 'https://www.gutenberg.org/files/174/174-0.txt', 'https://www.gutenberg.org/files/4300/4300-0.txt']

# >> TODO: Adjust the code of the mapper here
def mapper(key, value, n=3):
    """
    key ... url/filename
    value ... contents of the file

    yields a generator of palindromes, where the first entry in the tuple is a palindrom and the second entry is the count (1)
    """
    tokens = value.split()
    for i in range(0, len(tokens)):
        word = tokens[i]
        if len(word) == 1: # ignore "trivial" palindromes, i.e., words of length 1
            continue
        if word == word[::-1]:
            yield (word, 1)

def reducer(key, values):
    """
    key ... n-tuple
    values ... counts for the n-tuple
    """
    yield (key, sum(values))

# << TODO: End of code you have to change

if __name__ == "__main__":
    print("Running MapReduce for creating a language model...")

    # (1) Download books from Project Gutenberg and read their content
    files = {}
    try:
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for url in urls:
            tmp = urlparse(url)
            filename = os.path.basename(tmp.path)
            target_filename = os.path.join(tmp_dir, filename)
            print("- Downloading '%s' to '%s'." % (url, target_filename))
            if os.path.exists(target_filename):
                print("  File already exists, not downloading.")
            else:
                r = requests.get(url, allow_redirects=True)
                open(target_filename, 'wb').write(r.content)

            files[url] = open(target_filename, 'rt', encoding='utf-8').read()
    except:
        raise RuntimeError("Failed to download books: ", sys.exc_info()[0])

    # (2) Run mappers
    files = list(files.items())
    mapper_results = map(mapper, [x[0] for x in files], [x[1] for x in files])

    # (3) Gather results from mappers, sort and run reducers
    mapper_results = list(mapper_results)
    mapper_results_dict = {}
    for mapper_result in mapper_results:
        for key, value in mapper_result:
            if key not in mapper_results_dict:
                mapper_results_dict[key] = []
            mapper_results_dict[key].append(value)
    mapper_results_dict = mapper_results_dict.items()
    reducer_results = map(reducer, [x[0] for x in mapper_results_dict], [x[1] for x in mapper_results_dict])

    # (4) Gather restults form reducers and output them
    reducer_results = list(reducer_results)
    reducer_results = [list(x) for x in reducer_results]
    
    # rearrange reducer results and sort such that the most common n-grams come first
    reducer_results = [x[0] for x in reducer_results]
    reducer_results.sort(key=lambda x: x[1], reverse=True)

    print("Some reducer results. First 20 most common n-grams:")
    for i in range(20):
        print(reducer_results[i])
