# codding: utf-8
from nltk.corpus import wordnet as wn


def w2v_test():
    panda = wn.synset('panda.n.01')
    hyper = lambda s: s.hypernyms()
    print list(panda.closure(hyper))


class A:
    all_share = 0

    def __init__(self):
        self.a = 0

    def add_one(self):
        A.all_share += 1

    def output(self):
        print A.all_share

if __name__ == '__main__':
    # w2v_test()
    tmp1 = A()
    tmp2 = A()
    tmp1.add_one()
    tmp2.output()
