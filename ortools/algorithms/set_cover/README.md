# Set Cover

This folder serves as a temporary workspace to clearly highlight all files and modifications introduced during the collaboration between Francesco Cavaliere (c4v4) and the OR-Tools team.

The collaboration focuses on implementing the Set-Covering heuristic (hereafter CFT) described in:

    Caprara, Alberto, Matteo Fischetti, and Paolo Toth. 1999. “A Heuristic
    Method for the Set Covering Problem.” Operations Research 47 (5): 730–43.
    https://www.jstor.org/stable/223097

## Notes
This file also serves as a reference point for discussions on specific aspects of the implementation, especially when there is no better place to document them directly in the code.

- `namespace scp`: Most set-covering related algorithms and data structures use the `SetCover*` prefix. However, for clarity and improved readability, a dedicated namespace could be used.
- A key aspect of the CFT algorithm is managing the core vs. full model. The algorithm routinely operates on a small subset of columns (core model), refining the selection using reduced costs as a quality proxy. From past experience, correctly implementing the full/core model distinction is crucial to preventing complexity escalation in the implementation.

