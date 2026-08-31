# Paper Table 5: phonon-frequency RMSD status

Sources: paper DOI 10.1038/s41524-025-01611-8, ddmms/data mace-mof-0 release, and TU Graz DOI 10.3217/wyc7s-8en40.

## Decision

Skip Table 4 for now. It is useful as an out-of-sample force/energy/stress check, but the public `curated_test.zip` gives structures only; the DFT force/stress labels needed for an honest RMSD table are not present locally or in the ddmms release snapshot.

Prioritize Table 5. It is more directly tied to the paper's phonon claim. The reproduced-model row can be computed once DFT phonon band references from the TU Graz archives are available.

## Paper Table 5 values

Frequency RMSD is in THz. Values in parentheses are Gamma-only errors for the MACE models; the first number is the full-mesh error.

| MOF | MACE-MP-MOF0-v2 | MACE-MP-MOF0 | Gamma-only MTP | Gamma-only VASP MLP |
| --- | ---: | ---: | ---: | ---: |
| MOF-5 | 0.033 (0.085) | 0.039 (0.083) | 0.099 | 0.270 |
| UiO-66 | 0.082 (0.186) | 0.122 (0.198) | 0.111 | 0.312 |
| MOF-74 | 0.173 (0.427) | 0.243 (0.436) | 0.093 | 0.126 |
| MIL-53 | 0.138 (0.257) | 0.126 (0.239) | 0.156 | 0.264 |

## Local release assets already prepared

`data/mace_mof_0/results/phonon_nte_prep/` contains release band-frequency tables for foundation `0b`, published `mof0`, and published `mof0-v2` arms, plus POSCAR inputs for MOF-5, UiO-66, MOF-74, and MIL-53.

These are useful for sanity checks and plotting, but they are not DFT references. They cannot by themselves produce the Table 5 RMSD target.

## Reproduced model setup

The Table 5 runner now points at the successful multi-head reproduction:

`data/mace_mof_0/runs/mof0_paper_reproduce_force_mh_lr_20260829T124117Z/models/mof0_paper_reproduce.model`

Configured head: `pbe_d3`.

Protocol file:

`data/mace_mof_0/results/apples_to_apples/configs/phonon_protocol.yml`

Runner:

`data/mace_mof_0/results/apples_to_apples/scripts/run_phonons_apples.py`

## Current blocker for a real reproduced Table 5 row

The ddmms release provides DFT DOS curves, but Table 5 requires DFT phonon band/frequency references. Those are expected inside the TU Graz material archives under `DFT_phonon_band_structures`:

| required archive | size | md5 |
| --- | ---: | --- |
| `MOF-5.tar.gz` | 4.3 GB | `15c24738051921d1fe474fde3fa7e5f9` |
| `UiO-66.tar.gz` | 3.8 GB | `4b859edcbe072c9c5cc269ebca71d8d4` |
| `MIL-53_lp.tar.gz` | 6.8 GB | `4304c39b5343706a30ecdb14d6c9f148` |
| `MOF-74.tar.gz` | 17.6 GB | `20c202e107e4f842b591a69248632dd0` |

A previous probe downloaded only about 11 MB of `MOF-5.tar.gz` before SSL EOF, and the server returned `200 OK` to Range requests, so reliable resume was not available from that endpoint.

## Next executable step

Once at least `MOF-5.tar.gz` is fully available and md5-verified, extract only its DFT phonon-band reference, run the reproduced model phonon workflow for MOF-5, then compute frequency RMSD with:

`data/mace_mof_0/results/apples_to_apples/scripts/compute_frequency_rmsd.py`

After MOF-5 validates the adapter, repeat for UiO-66, MOF-74, and MIL-53.
