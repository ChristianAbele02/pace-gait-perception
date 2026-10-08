# Third-party notices

This repository bundles third-party code and art so the Unity project builds and runs as a
unit. Those components are **not** covered by this repository's own licence (see `LICENSE`);
each remains under the terms of its original author or vendor, listed below.

If you reuse this repository, it is your responsibility to check that your use is compatible
with each of these. Where a component's licence does not permit redistribution for your
purpose, remove it and obtain it from its original source instead.

---

## Unity engine components

| Path | Component | Origin / terms |
|---|---|---|
| `Assets/Standard Assets/` | Unity **Standard Assets** | © Unity Technologies. Distributed under the [Unity Companion License](https://unity.com/legal/licenses/unity-companion-license) / the Asset Store EULA that shipped with Unity 2017.4. |

No compiled player builds are included. A build made from this project contains the Unity
runtime and the Mono class libraries, which stay under Unity's and the Mono Project's terms.

## CAVE, tracking and hardware integration

| Path | Component | Origin / terms |
|---|---|---|
| `Assets/UniCave/` | **Uni-CAVE** — Unity plugin for non-head-mounted VR display systems | Living Environments Laboratory, University of Wisconsin–Madison. See Tredinnick, Boettcher, Smith, Solovy & Ponto (2017), *IEEE VR*, [doi:10.1109/VR.2017.7892342](https://doi.org/10.1109/VR.2017.7892342). Upstream: <https://github.com/widVE/UniCAVE>. |
| `Assets/UniCave/Plugins/` (`unityVrpn.dll`, `libunityVrpn.so`, `unityVrpn.bundle`) | **VRPN** native binding (Virtual Reality Peripheral Network) | VRPN is released into the public domain by the original authors (University of North Carolina at Chapel Hill); bundled binaries are builds of that library. Upstream: <https://github.com/vrpn/vrpn>. |
| `Assets/OptiTrack/` | **OptiTrack Unity Plugin** (incl. `NatNetLib`) | © NaturalPoint, Inc. Redistributed under the plugin's own vendor terms — see `Assets/OptiTrack/README_OptiTrack_Unity.txt` and NaturalPoint's licence. **Not** covered by this repository's licence. |
| `Assets/Scripts/Optitrack/` | NatNet stream parsing (`FrameData`, `DataDescription`, `ParsingUtils`) | Adapted from NaturalPoint's NatNet sample code for this rig; retained from earlier Interactive Locomotion Lab work. |

The native hardware bridges the rig runs alongside Unity (`runTreadmillServer` for the Modbus
treadmill, `runMotionBaseServer` for the Moog motion base) and the render nodes' signed
start/stop listener belong to the Interactive Locomotion Lab installation, Bielefeld
University. They are **not** included here.

## Art assets

| Path | Component | Origin / terms |
|---|---|---|
| `Assets/NatureStarterKit2/` | **Nature Starter Kit 2** | Free Unity Asset Store package by *Shapes* (Art Director Andras Nagy; Technical Director Zsolt Torok). The publisher states the contents may be used in commercial products — see `Assets/NatureStarterKit2/info.txt`. Redistribution remains subject to the Unity Asset Store EULA. |
| `Assets/Textures/terrain/` | Terrain textures | See `Assets/Textures/terrain/about.txt` for the per-file attribution that shipped with them. |
| `Assets/Sounds/` | Audio cues | Retained from earlier Interactive Locomotion Lab work. |

## Code adapted from published tutorials

| Path | Component | Origin / terms |
|---|---|---|
| `Assets/Scripts/Splines/` | Cubic Bezier spline classes and their editors (`Bezier`, `BezierCurve`, `BezierSpline`, `SplineWalker`, `Line`, and the matching `Editor/` inspectors) | Derived from Jasper Flick's **Catlike Coding** "Curves and Splines" Unity tutorial (<https://catlikecoding.com/unity/tutorials/curves-and-splines/>), then extended for this project with an arc-length parameterisation and 3-D offset accessors. Catlike Coding's tutorial code may be used in your own projects; the tutorial text itself is © Jasper Flick. |

## Prior work at the Interactive Locomotion Lab

Parts of the Unity project — the path/terrain generation, the logging controller, the motion-base
and treadmill controllers, and the OptiTrack receiver — originate in the lab's earlier obstacle-
traversal ("hurdling") study and were inherited with the rig, then modified for this experiment.
That work is documented in:

- **Hofmann, F. (2023).** *Human Motion Planning and -Adjustment during Obstacle Traversal in
  Virtual Reality.* PhD thesis, Faculty of Biology, Bielefeld University.
- **Hofmann, F. & Dürr, V. (2022).** When running is easier than walking: effects of experience
  and gait on human obstacle traversal in virtual reality. *Experimental Brain Research*
  240(10):2701–2714. [doi:10.1007/s00221-022-06443-2](https://doi.org/10.1007/s00221-022-06443-2)

Some source files still carry the original authors' comments and identifiers. Vestiges of the
predecessor paradigm remain in a few places (for example the Unity `productName` is still
`Hurdling Demo`, which determines the render nodes' log path).

## Reference data

`Analysis/davidson_power/` derives this study's power analysis from the openly published data of:

- **Davidson, M. J., Verstraten, F. A. J. & Alais, D. (2024).** Walking modulates visual
  detection performance according to stride-cycle phase. *Nature Communications* 15:2027.
  [doi:10.1038/s41467-024-45780-4](https://doi.org/10.1038/s41467-024-45780-4)

Their raw `.mat` files are **not** redistributed here; the scripts read them from the authors'
own OSF deposit. Only derived summary statistics and figures are committed.
