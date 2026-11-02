# Online play: the host-authoritative pattern

Pattern notes from Sandblood's online 1v1 (ENet over UDP, Windows/Mac/Linux/Android in any combination, an online
service for quick match and a leaderboard). Use them as the starting contract for any physics game with a few
players per match. Write the project's own `docs/online_brief.md` from this before any lobby UI.

## Model

Godot physics (Jolt included) is not deterministic across machines, and not even run to run (contact order is
thread-dependent), so lockstep and rollback are out for physics games.

- **The host simulates everything** exactly as offline: its own side from its devices, each remote side from that
  client's input stream (a remote "brain" that pops an input buffer).
- **The client simulates nothing.** It builds the same match from the same seed and loadouts (same bodies, art,
  equipment), switches its simulation off, makes every body kinematic and poses it from host snapshots, a little in
  the past, interpolated. Hits, effects events and the result come from the host as events.
- **Predict only what is cheap and safe**: for example the player's own aim cursor, computed locally with the same
  maths the host uses. Predicting a ragdoll means simulating it and correcting it, which jitters under divergence.
- Keep the networking code split like the rest of the game: pure `RefCounted` logic (config, quantisation, snapshot
  and event codecs, input packer, input buffer, interpolation clock, lobby messages, bandwidth meter, fake link) with
  unit tests, and thin nodes (session/peer, host driver, client driver, world adapter, lobby screen).

## Messages

Every packet starts with a type byte; one channel; unreliable-ordered or reliable by transfer mode.

| Message | Dir | Mode | Rate | Payload |
| --- | --- | --- | --- | --- |
| HELLO | C->H | reliable | once | protocol version, content hash, name, loadout |
| LOBBY | H->C | reliable | on change | the same plus accepted / reason |
| START | H->C | reliable | per match | seed, loadouts, rules, names |
| INPUT | C->H | unreliable ordered | 60 Hz | newest tick, then the last N ticks of input (redundancy) |
| SNAP | H->C | unreliable ordered | 30 Hz | host tick, last applied client input tick, per-body state |
| EVENTS | H->C | reliable | when things happen | tick-stamped hit/effect events, key mask + present values |
| OVER / LEAVE | both | reliable | once | result, or reason |

- **Version and content hash.** Lobby messages carry the protocol version and a hash of every data file (re-written
  with sorted keys so line endings do not matter) plus every `Tuning` value. Refuse a mismatch with a readable reason
  on both sides. `bytes_to_var` lobby messages with objects refused and a size cap.
- **Input redundancy.** Each INPUT packet carries the last 4 ticks, so one packet fills a gap of 3 lost before it.
  The host's input buffer plays ticks in order about 1 tick behind the newest, repeats the last input for a missing
  tick, skips back when it grows too deep, and adapts (an underrun grows it, a quiet window shrinks it).
  Press-once buttons on a skipped tick carry onto the next tick played. After about 12 quiet pops, movement and
  buttons go neutral (otherwise a dropped client keeps walking until the timeout).
- **Snapshots.** Quantise: position 3 x i16 at 1 mm, rotation smallest-three in 32 bits, linear velocity 3 x i16 at
  1 cm/s; send no angular velocity if the client never integrates rotation. Example size: 522 bytes for two ragdolls
  of 14 bodies each, about 17 KB/s at 30 Hz. Send the engine time scale so slow motion matches.
- **Events.** A fixed key table, booleans in a mask, zeros left out, halves for floats. The client queues each event
  by tick and re-emits it on the local signal when its render clock reaches that tick, so effects and sound fire in
  step with the picture. **Split packets by bytes, not record count**: relays drop packets over their size limit
  silently (one had a 1164-byte limit).

## Budget

Count 42 bytes overhead per packet (IPv4 20 + UDP 8 + ENet about 14) and write a per-direction table (target under
30 KB/s). Sandblood measured about 18 KB/s host to client and 5 KB/s client to host. Delta compression against the
last acked snapshot halves it if mobile data matters.

## Latency handling

- The client's render clock runs in host ticks, a delay behind the newest: one snapshot interval + a margin + 1 x
  measured arrival jitter (0.5 x made the clock outrun the stream on a lossy link), clamped, eased at up to +-10%
  speed, jumping only on a large error, never backwards. Restart it from the stream when several snapshots in a row
  arrive behind it.
- Blend bodies with a cubic Hermite curve through both snapshots' velocities. Past the newest snapshot, extrapolate
  by velocity for a few ticks with a fading velocity and the last turn rate, then hold.
- **Switch ENet's packet throttle off** for unreliable traffic on connect (deceleration 0): a frame hitch looked like
  congestion and ENet dropped half the snapshots on a clean link. Flush right after each send.
- **Never close a peer from its own signal**; defer the close to the next idle frame (closing inside `poll()` crashed).
- Measure input-to-screen as the newest input index sent minus the input index the shown pose had applied.
- Timeouts with reasons: a few seconds without a packet in a match, longer while connecting.
- A pause menu in an online match sends neutral input and does not pause the tree.

## Soak and tests

- Unit tests: quantisation error bounds, codec round trips and sizes, input redundancy under 20% loss, the input
  buffer, the interpolation clock under jitter, lobby messages, the fake link, the byte meter.
- **A fake link** (`--net-lag=<ms> --net-jitter=<ms> --net-loss=<0..1>`) on what this machine sends.
- **Loopback sim in the suite:** host and client in one process (the client match in a SubViewport with its own
  World3D), real sockets on 127.0.0.1, fake 50 ms + jitter + 5% loss, a short match; checks lobby, inputs reach the
  host, budget, same result, client state equals host state at the same tick within quantisation, every event plays.
- **Soak tool** (`tools/net_soak.sh --matches=N --profiles="60:0.02 150:0.03 250:0.05"`): bot vs bot over real ENet per
  latency/loss profile; prints snapshot rate, interpolation delay, resyncs, pose error mean/p95/max per body group,
  input to screen, packet sizes, bandwidth, same result both ends. Record the reference pose at the tick the
  snapshot is stamped with: a one-tick shift tripled every error in the first soak.
- A drop sim covers leave, vanish, stall and a protocol mismatch.

## Service layer (matchmaking, ratings, leaderboard)

- Menus talk to one `OnlineService` interface (login, quick_match, cancel, report_result, request_leaderboard,
  my_rating); each backend is a subclass: a real service (EOS, Steam, or your own server), a LAN backend (UDP
  broadcast, also to 127.0.0.1 so two instances on one machine meet) and a scriptable stub for tests.
- Any `MultiplayerPeer` (ENet, a relay peer, Steam sockets, WebRTC) can carry the session if the session code uses
  only the packet API.
- NAT: v1 can be direct IP (LAN, forwarded port, VPN mesh). Across home routers you need a rendezvous plus UDP hole
  punching and a relay fallback for symmetric and carrier-grade NAT, or a service that does both.
- **The host can lie.** Each side reports the result from its own view, HMAC-signed with a per-match key; disagreeing
  reports are not rated; leaving loses; very short matches are void; damp repeat pairs. A client-written stat can
  always be forged: a serious leaderboard needs a small verifier service that holds the server credentials.
- Mock a plugin's native singleton to run its real GDScript in tests without the binaries.

## Credentials

- Client credentials (for example an online service's client id and client secret) only name a client policy and
  ship inside builds, so the policy must allow nothing dangerous. They still never go into git, docs or memory.
- Keep them in a git-ignored `config/<service>_credentials.cfg` (the kit's `.gitignore` covers `config/*credentials*`).
  The game looks in `res://config/`, then next to the executable, then `user://`. Export scripts copy the file next to
  the desktop build; an APK has no folder beside it, so the Android preset's `include_filter` must name the file
  (`tools/export_android.sh` warns when it does not). Without the file the game falls back to the LAN backend.
- Server credentials live only on the verifier.
