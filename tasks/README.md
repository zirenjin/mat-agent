# tasks/

Task specifications. Each `tasks/<task_id>/task_card.md` constrains a scientific
*question* (data, compute, evaluator, integrity, baseline) and leaves the
*solution* open.

| Task | Kind | Card |
| --- | --- | --- |
| `mace_mof0_discovery` | autonomous research: beat the strongest human MOF-potential baseline on MOF lattice dynamics | [task_card.md](mace_mof0_discovery/task_card.md) |

The MACE-MP-MOF0 *reproduction* walkthrough remains at
`docs/task_cards/mace_mof0_reproduction_task_card.md`. It is a scripted
reproduction, not an open research task; the discovery task reuses its
infrastructure (data release, scorers, Docker image) as frozen
`benchmark/` components.
