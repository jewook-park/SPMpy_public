# SPMpy

---

## 🔓 Public Release Update — 2026-01-09

On **2026-01-09**, a **curated subset of SPMpy** was intentionally released to a
separate public repository (**SPMpy_public**) as part of a controlled code-sharing
process.

### Key points of this release
- This public release originates from the private development branch  
  **`STMgroup_release`**.
- The public repository is **not a fork** and does **not expose the full development history**.
- Only code deemed appropriate for external visibility was included.
- Ongoing development, experimental features, and internal utilities
  remain exclusively in this **private repository**.

This approach is designed to:
- enable limited external code sharing,
- preserve internal development flexibility, and
- comply with ORNL and DOE software release and security policies.

Future public updates, if any, will be performed **manually and selectively**
from designated release branches.

---

**Authors**: Dr. Jewook Park, CNMS, ORNL  
Center for Nanophase Materials Sciences (CNMS), Oak Ridge National Laboratory (ORNL)  
Email: parkj1@ornl.gov

---

## Overview

SPMpy is a collection of Python functions for analyzing multidimensional scanning probe
microscopy (SPM) data, including **STM/S and AFM**.

It leverages advances in computer vision and the scientific Python ecosystem
for efficient data processing and visualization, inspired by tools such as
[Wsxm](http://www.wsxm.eu/), [Gwyddion](http://gwyddion.net/), and
[Fundamentals in Data Visualization](https://clauswilke.com/dataviz/).

SPMpy is developed and maintained by **Jewook Park (SPMpy)**.

Contributions, suggestions, and feedback are welcome in both **Korean and English**
via [GitHub](https://github.com/jewook-park/SPMpy) or
[email](mailto:parkj1@ornl.gov).

---

## Repository Status (Internal Use)

This repository was converted from **public** to **private**.

SPMpy is currently under **active research and development**, and the full development
history (including experimental code and intermediate implementations)
is not intended for public exposure at this stage.

Access is limited to **STM group members at CNMS**.

Access to this private repository is granted by invitation.
Please accept the GitHub invitation before cloning.

---

## _For STMers in the CNMS STM Group_

SPMpy is intended to be tested and improved through **STM data analysis**.

STM group members are encouraged to:

- Load and analyze **their own STM/STS datasets** using SPMpy
- Identify and report:
  - unexpected behavior
  - missing functionality
  - unclear interfaces
  - performance or visualization issues
- Provide feedback, bug reports, or suggestions

This feedback is essential for improving the **robustness, usability,
and scientific reliability** of the codebase.

---

## Data Handling

- Uses [**Xarray**](https://docs.xarray.dev/) as the primary data container.
- Supports conversion of **Nanonis Controller (SPECS)** datasets into Xarray using
  [nanonispy](https://github.com/underchemist/nanonispy).
- Enables simultaneous manipulation of multiple channels while preserving metadata
  (headers).
- Other SPM controller formats can be analyzed once converted into Xarray.
- ~~2D images from **Gwyddion (`*.gwy`)** can also be imported for extended analysis.~~

---

## Environment

- The development and testing environment is based on
  [**Miniforge (mamba)**](https://github.com/conda-forge/miniforge).
- Please refer to the **previously shared environment setup file/document**
  for detailed installation and dependency information.
- Updated on **2026-01-05** by Jewook Park for internal review in the CNMS STM group.

---

## License

This software was developed at **Oak Ridge National Laboratory (ORNL)**.

The licensing terms have **not yet been determined** and will be established
in accordance with ORNL and DOE policies.

At this stage, the repository is provided for **internal and collaborative review only**.
Once an official license is approved, this section and the LICENSE file
will be updated accordingly.

---

## Usage Policy

- Please **clone this repository directly** using Git.
- Do **not redistribute** the repository or create public forks.
- This codebase is provided for **internal research use only**.
