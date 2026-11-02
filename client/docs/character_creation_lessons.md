# Humanoid character creation: lessons

These come from building the Skate Park lead skater, an adult anime surf-skater made from code-built meshes, on 2026-10-21. It took four rounds to reach a look the owner liked. Most of the misses could have been caught before the owner ever saw them.

## What went wrong

| # | What happened | Root cause | Do this instead |
|---|---|---|---|
| 1 | Round 1 came back "not good enough": a generic, primitive look. | The brief was "make her beautiful", with no reference material and no numbers. | Research real references first (anime proportion guides, VRoid defaults, the shipped games the owner likes). Write a numeric spec of landmark heights, widths and depths, and the face layout, before any modelling. |
| 2 | The character was about 1.25 m tall next to a real-size 0.8 m skateboard. | The skeleton constants (leg and arm lengths) were frozen to protect the animation rig, so the body was built to fit the rig rather than the person. | Set the target height and head count first (adult anime is about 7 to 7.5 heads), then size the skeleton to match. Animation code must read the skeleton constants, never hard-coded heights, so resizing is safe. |
| 3 | A "barrel chest". | The torso was one circular lathe, so width equalled depth everywhere. | Loft elliptical cross-sections, each with its own width, front depth and back depth. The ribcage should be about 70% of the hip width, with a flat back. Build the bust as two separate forms, not a wider cylinder. |
| 4 | The head read wrong: eyes too far apart on a flat face, stern brows, helmet hair. | The face was a featureless lathe with a scaled head node. The brows were angled at the inner end. The hair was one cap with a few planks. | Build the head at real size: about 0.225 m chin to crown for a 1.62 m adult. Put the eyes one eye-width apart at or just below mid-head, on a curved face. Use thin, low, softly arched brows. Build the hair from 20 to 30 tapered, overlapping locks over a cap. |
| 5 | The bust sat too low and clipped. The bikini bottoms clipped into the legs. | Reviewed only in the rest pose, front view. Garments were placed by eye, not offset from the skin. | Build every garment as an offset from the skin's own section data. Check a full pose grid (idle, crouch, push, tuck, grabs, walk, sit) from the front, side, three-quarter and below before calling a model done. |
| 6 | Shoes and boards looked generic. | They were modelled from memory. | Research the real product's anatomy first (shoe: foxing, toe cap, eyelets, collar; truck: baseplate, kingpin, bushings, hanger), then loft from measured cross-sections. |
| 7 | Head, body, shoe and board workers broke each other's tests mid-run, and parallel edits nearly collided in shared files. | Several workers shared placeholder_skater.gd and one test file. | Give each part its own file and builder (head, body, garment per slot, shoe, board), with a written contract for attachment points and palette keys. Shared files take only small targeted edits. Tests for each part live in that part's own test file. |
| 8 | The customiser stopped recolouring the shoes and hair after the rebuilds. | The new builders invented new palette keys, and nobody updated the mapping. | Keep the palette keys in one table. A builder that adds keys must add them to that table and to the customiser mapping in the same change. Add a test that every catalog colour slot changes a visible material. |
| 9 | The art direction changed mid-build: kid-friendly became a Teen or M rating. | The rating and tone were never pinned down. | Ask for the target rating and tone on day one and write it into memory and the art spec. It decides outfits, proportions, physics and violence. |

## Checklist for a new humanoid character

1. Ask for the rating, tone, reference images, and the two or three games the look should sit beside.
2. Write the spec: height and head count; landmark heights; section widths and depths; the face layout; the hair lock plan; the palette with hex values.
3. Size the skeleton to the spec. Check that the animation code reads only skeleton constants.
4. Use a base body plus layers: one smooth, featureless mannequin base body per body type (no anatomical detail). Every garment is a separate layer lofted outside it, and each garment hides the body regions it fully covers (masking), so skin never pokes through.
5. Build the parts in separate files: head, hair (registry of styles), body (elliptical lofts), garments (one file per slot, offset from the skin sections), shoes, held props.
6. Add sway pivots for secondary motion from the start (hair, ponytail chains, soft body parts, loose clothing), named by prefix.
7. Render a standard sheet every iteration: front, side, three-quarter, back, face close-up, plus the full pose grid. Look at every image before reporting.
8. Write tests for proportions (height, head count, ribcage narrower than hips, eye spacing), feet on the ground, no pure-green debug colours, garment clearance, and the triangle budget.
9. Keep the palette keys in one table and the customiser mapping in sync.
10. Show the owner a side-by-side against the references, then ask for notes on face, body and outfit separately.

## Round two: why we threw the code-built character away (2026-10-22)

After six more rounds (code lofts, a Blender rebuild, a Houdini KineFX body, three polish passes) the owner said the character had cost far too many tokens and must be remade one last time. The real mistake was the approach, not any single round.

| # | What happened | Root cause | Do this instead |
|---|---|---|---|
| 10 | Every round fixed one complaint and exposed the next (face, bust, clipping, hands, collars, straps). | The body, face, hair and clothes were generated by code or by agents describing meshes in numbers. Agents can't see well enough to sculpt, so quality plateaued at "placeholder". | Never sculpt a humanoid in code. Start from a character made in a real character creator (VRoid Studio, Character Creator, MetaHuman, MakeHuman) or a licensed base model. Code only adapts it: bone mapping, materials, recolours, attachments. |
| 11 | Three toolchains (code, Blender, Houdini) were each wired into the game in turn. | Each new tool was tried as a fix for quality, and each needed its own bake, rig mapping and tests. | Pick the asset pipeline once, on day one, and spike it end to end (import, rig, one pose, one render) before building anything on top. |
| 12 | Late discovery that the cloud proxy blocks many asset sites (opengameart, VRoid Hub, GitHub releases). | Asset sourcing was never tested. | On day one, test-download one file from each asset source the project needs. If blocked, fetch through the owner's machine (Remote Control) into an asset branch. |
| 13 | A custom 17-joint rig with procedural poses meant every new body needed a hand-written adapter and retuned poses. | The rig was invented before any real character existed. | Use the standard humanoid bone set (VRM / Unity Humanoid / Mixamo names) from the start, so any creator model drops in. Keep procedural posing, but drive it through standard bone names. |
| 14 | Wardrobe garments had to be re-lofted for every body change. | Clothes were tied to one body's section data. | Use the creator's own outfit system, or garments skinned to the same standard skeleton, so a body swap doesn't break clothes. |
| 15 | Houdini body polish was thrown away by the game's loader, costing a whole round. | The asset was built before anyone read the game code that consumes it. | Before building an asset the game will use, read the importer or loader code and check which data survives it. |
| 16 | Hand-built meshes (a fountain wall, a road) were inside out but looked fine in previews. | Preview renders draw both sides of every face. | Check face direction numerically (normals against the outward direction) on any hand-built mesh. |
| 17 | Wrong guesses about tool conventions (tube radius order, rotation sign, menu tokens) each cost a full rebuild. | Conventions were assumed, not tested. | Confirm each convention once with a ten-line test before using it in a real build. |
| 18 | Two commit hashes were misquoted in reports. | Hashes were reported from memory before the push finished. | Report a commit hash only after the push has returned it. |

### Rule of thumb

If the owner says a character "looks like a placeholder" twice, stop iterating. Switch to a creator-made or licensed model the same day.
