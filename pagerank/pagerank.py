import os
import random
import re
import sys
import numpy

DAMPING = 0.85
SAMPLES = 100000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    probability = {}

    anti = 1 - damping_factor

    for i in corpus.keys():
        probability[i] = anti / len(corpus)

    if len(links) == 0:
        # return equal probability
        for i in corpus.keys():
            probability[i] = 1 / len(corpus)
        return probability

    for link in links:
        probability[link] += damping_factor / len(links)

    return probability


def sample_pagerank(corpus, damping_factor, n):
    # Finished
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRank = {}
    samples = []

    samples.append(random.choice(list(corpus.keys())))

    for i in range(n):
        probability = transition_model(corpus, samples[i], damping_factor)
        nextPage = numpy.random.choice(list(probability.keys()), p = list(probability.values()))
        samples.append(nextPage)
    
    for page in corpus.keys():
        pageRank[page] = samples.count(page) / len(samples)
    return pageRank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pageRank = {}

    for page in corpus.keys():
        pageRank[page] = 1 / len(corpus)

    n = len(corpus)
    d = damping_factor

    algConst = (1 - d) / n

    while True:
        prevRank = pageRank.copy()
        for page in pageRank:
            sumAlg = 0 # Assumes no incoming links
            for link in corpus.keys():
                if len(corpus[link]) == 0:
                    corpus[link] = corpus.keys()
                if page in corpus[link]:
                    iterative = pageRank[link] / len(corpus[link])
                    sumAlg += iterative
            pageRank[page] = algConst + d * sumAlg
        # break out if error is less than .001
        max_change = -10000
        for page in pageRank:
            if abs(prevRank[page] - pageRank[page]) > max_change:
                max_change = abs(prevRank[page] - pageRank[page]) 
        if max_change < 0.001:
            break
        
    return pageRank

if __name__ == "__main__":
    main()