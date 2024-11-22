import math

from load_melons import load_melons as ld
from load_melons import load_tests as ld_tests
from BayesClassifier import BayesClassifier
import numpy as np


def main():
    datas = ld()
    bc = BayesClassifier(datas)
    tests = ld_tests()
    tests = np.array(tests)
    tests = tests[:, 1:]
    for test in tests:
        pGood = bc.get_Post(test, '是')
        pBad = bc.get_Post(test, '否')
        ans = "好" if pGood > pBad else "坏"
        print(f"feature:{test} PGood:{pGood} PBad:{pBad} {ans}")

main()
