# Architecture {#overview}

\tableofcontents

## Project Overview

The LLZK project consists of three main components:

1. **Frontends**, which translate a source ZK language to LLZK.
2. **Passes**, which analyze or transform the IR.
3. **Backends**, which process LLZK into another destination format (e.g., R1CS constraints) or analyze the IR to identify bugs or verify properties of the source language.

The general workflow of using LLZK is therefore as follows:
1. Translate the source language into LLZK using [a frontend tool](\ref frontends).
2. Use the [llzk-opt tool](\ref llzk-opt) to perform any transformations using LLZK's [pass infrastucture](\ref pass-overview).
3. Provide the transformed IR to a [backend](\ref backends) for further analysis or use.

### Frontends {#frontends}

Frontends are currently not contained within the LLZK repository, but are rather
maintained in separate repositories, using LLZK-lib as a dependency.

Veridise currently maintains the following frontends:
- [Zirgen](https://github.com/Veridise/zirgen-to-llzk)
<!-- TODO: Update this link to a doxygen site at some point. -->

For information on how to create a new frontend, please refer to the \ref translation-guidelines.

### Passes {#pass-overview}

LLZK provides three types of passes:
1. *Analysis* passes, which compute useful information about the IR used to implement other passes or backends.
2. *Transformation* passes, which restructure or optimize the IR.
3. *Validation* passes, which ensure the IR has certain required properties.

User documentation about how to use these passes is provided in \ref tools.

Developer documentation can be found:
- In the Analysis directories:
    - General, multi-dialect analyses: \ref include/llzk/Analysis, \ref lib/Analysis
- In the Transforms directories:
    - General, multi-dialect transforms: \ref include/llzk/Transforms, \ref lib/Transforms
    - `array` transforms: \ref include/llzk/Dialect/Array/Transforms, \ref lib/Dialect/Array/Transforms
    - `polymorphic` transforms: \ref include/llzk/Dialect/Polymorphic/Transforms, \ref lib/Dialect/Polymorphic/Transforms
- In the Validators directories
    - General, multi-dialect validators: \ref include/llzk/Validators, \ref lib/Validators

### Backends {#backends}

Built-in backends will be added to the [llzk-opt tool](\ref llzk-opt) as they are developed.
Currently, LLZK provides no built-in backends, but an R1CS backend is current in the works.
Veridise also plans to release several analysis backends based on prior tooling (namely [Picus][picus-v2] and [ZK Vanguard][zk-vanguard]), which will allow Veridise to provide automated verification and analysis for any ZK language that has an LLZK frontend.

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref mainpage | \ref setup |
</div>

[picus-v2]: https://docs.veridise.com/picus-v2/
[zk-vanguard]: https://docs.veridise.com/zkvanguard/
