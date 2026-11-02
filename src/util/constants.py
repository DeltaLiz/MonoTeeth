"""Dental part order shared by data, queries and visualization."""

import numpy as np

# DMM part ids per jaw: gum (0), then FDI teeth without wisdom teeth.
UPPER_IDS = (0, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27)
LOWER_IDS = (0, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47)
JAWS = (("upper", UPPER_IDS), ("lower", LOWER_IDS))
# Query k, semantic channel k and latent row k all refer to PARTS[k].
PARTS = tuple((jaw, i) for jaw, ids in JAWS for i in ids)

_JAW_PALETTE = [
    (255, 82, 0), (255, 31, 90), (249, 255, 33), (30, 42, 120),
    (255, 23, 0), (255, 142, 0), (255, 228, 0),  # x1-x7
    (252, 199, 44), (95, 113, 97), (0, 161, 171), (199, 0, 57),
    (243, 113, 33), (192, 226, 24), (8, 60, 90),  # x1-x7 of the other side
]
PALETTE = np.array([(243, 197, 197), *_JAW_PALETTE, (143, 97, 97), *_JAW_PALETTE], dtype=np.uint8)
