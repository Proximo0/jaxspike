# New project checklist

From the build guide: https://claude.ai/artifact/WtmrkVRtizP8oJBRwhxdWW. The steps behind each line are in
`docs/playbook.md`.

Day one (before any content):
- [ ] Vision, rating and tone, v1 scope and 3-6 measurable feel targets agreed with the owner and written to memory.
- [ ] Debug layer is in: `Flags`, frame stamp (F10), render toggles, `Tuning` for every feel number.
- [ ] CI is on: push to GitHub and confirm the `checks` workflow runs unit, lint and sims green.
- [ ] `CLAUDE.md` filled in (what the game is, layout, rules) and `docs/agent_brief.md` units and contracts written.
- [ ] Agent dashboard is up (`docs/agent_dashboard.md`): agents, models, token pull per minute, CI. Refresh it on
      every worker hand-back.
- [ ] Session-start hook works in a cloud session (Godot pinned to 4.7.2, project imported).
- [ ] Asset pipeline spiked end to end (import, rig, one pose, one render) and one file test-downloaded from each
      asset source the project needs.

Before building systems:
- [ ] Feel targets agreed with the player in words and numbers (jump height, top speed, turn radius, timing windows),
      written into `docs/` and into `Tuning` defaults. Check a lab WARN against these before tuning it away.
- [ ] Piece collision rules written down before building pieces (minimum collision edge size, nothing coplanar,
      side walls below ground, lift visual meshes, never colliders).
- [ ] Every logic module has a unit test; every feel-critical behaviour has a sim.
- [ ] Each build step has a contract brief (`docs/<step>_brief.md`) before workers fan out.

While building:
- [ ] Reproduce before fixing: a test, sim, bot run, replay or screenshot sampled for pixels. If it cannot be
      reproduced, ship toggles instead of guessing.
- [ ] Fix the traced mechanism. Check git history for when a bug first appeared.
- [ ] A row goes into `docs/lessons_learned.md` whenever something goes wrong.

Once there is a level:
- [ ] A play-bot (`tests/bot/bot.tscn`, shardable with `--shard=i --nshards=n`) that plays it and flags issues;
      `tools/check_all.sh` picks it up automatically. Fan it out in `.github/workflows/checks.yml`.
- [ ] A lab/sweep scene (`tests/lab/lab.tscn`) for measuring snags and feel numbers, if the game has physics.
- [ ] A perf harness on a scripted worst case, with budgets per quality level (`docs/performance.md`).

Characters:
- [ ] Rating and tone agreed (kid-friendly, Teen, M), plus reference images and comparable games.
- [ ] Before building any humanoid, follow `docs/character_creation_lessons.md`: start from a creator-made or licensed
      model on a standard humanoid skeleton; never sculpt a humanoid in code.

Online and mobile (if in scope):
- [ ] Online: write the host-authoritative contract and a fake-link soak before the lobby UI (`docs/online.md`).
      Credentials only in git-ignored `config/*credentials*`.
- [ ] Android: `tools/export_android.sh` builds a debug APK on day one of mobile work; touch controls and a low
      quality level are planned with it (`docs/android.md`).

Running the project:
- [ ] After the bot exists, run the lean loop in `docs/production_pipeline.md` (measure, fix 1-2 traced defects, verify on a clean clone, push).
- [ ] Lead verifies every worker hand-back on a clean clone before pushing; builds only at milestones.

Shipping:
- [ ] `tools/check_all.sh` green and CI green, then `tools/export_windows.sh` (and `tools/export_android.sh`).
- [ ] A 50+ match soak on real hardware found no per-match hitches.
- [ ] Lead-only git: stage by name, one commit per integrated step.
- [ ] Lessons harvested back into the starter kit (generic rows to its `docs/lessons_learned.md`, stories to
      `docs/lessons/<game>.md`).
