"""Row: Retrieved documents / MultiDoc2Dial.

Grounded document-dialogue turns; each user turn's referenced documents form a
snapshot. Deduplicated to one dialogue per task. Treatment: none.

Published: Prefix 56.01%  PIC 39.97%  PIC-proc 39.97%  in-sess 3.04  cross-sess 36.92
Data: local data/multidoc2dial/multidoc2dial_{doc,dial_train,dial_validation}.json
"""
import _bootstrap
from prefix_disjoint import measure_disjoint
from rag500 import sessions_multidoc2dial


def main():
    m2 = sessions_multidoc2dial()
    seen = set()
    m2 = [x for x in m2 if not (x["task"] in seen or seen.add(x["task"]))]
    d = measure_disjoint(m2)
    _bootstrap.emit("Retrieved documents / MultiDoc2Dial",
                    {"prefix": 56.01, "pic": 39.97, "pic_proc": 39.97,
                     "in_sess": 3.04, "cross_sess": 36.92}, d)


if __name__ == "__main__":
    main()
