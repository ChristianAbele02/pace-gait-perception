# Operator guide: running the PACE experiment

How to run one participant from arrival to backed-up data, written for an operator who did not
build the paradigm. For the science, see [`README.md`](README.md). To set the project up on
another installation, start with [§12](#12-one-time-rig-setup).

**Language.** The guide is in English. The **participant briefing script in
[§3](#3-welcoming-and-briefing-the-participant) is given in German and English**. The two are
direct translations, so read the participant's version verbatim.

> **The experiment in one paragraph.** The participant stands, or walks in place on a
> treadmill, inside the CAVE and performs a two-alternative forced-choice tilt task. A striped
> patch (Gabor) flashes for 100 ms, tilted slightly left or right, and they press Left or
> Right for the direction it leaned. A short QUEST+ calibration sets difficulty individually,
> so most trials sit near that person's threshold. Each participant comes **twice**.

---

## 0. The five things that must be right

Each of these silently destroys data if wrong.

1. **Block = condition.** `0` = standing, `1` = walking. Block controls nothing else.
2. **Trial = run number.** Every block gets its **own** Trial number. Reusing one overwrites
   nothing, but the re-run is stored under an odd run number, and only your notes say which
   block it is. Follow the tables in [§6](#6-run-procedure-exact-values-to-type).
3. **Visit = 1 or 2.** It drives the condition-order crossover. A wrong value undoes the
   counterbalance and cannot be repaired afterwards. The setup screen will not start a
   calibration or block until the Visit is typed.
4. **Debug toggle OFF.** Debug mode disables logging, so you get no data at all.
5. **Back up the raw logs before you leave the lab.** They are irreplaceable.

### Safety

- Before the participant steps on, show them the **emergency stop** and confirm the
  experimenter can reach and use it at any time. The e-stop is an **independent circuit**,
  isolated from the experiment software. It works even if Unity has hung. Mechanical brakes
  halt the belt **and** the motion base in **under 10 ms**. Buttons sit at the **operator
  workstations** and at the **laboratory exit**. Know where the nearest one is before you start.
- **Remove the boarding staircase** once the participant is on the platform. Put it back
  before they step off.
- Ask them to report dizziness, nausea or unsteadiness immediately. Moving visual
  surroundings plus walking in place can cause motion sickness in a minority of people.
- Stop the block rather than "pushing through" a complaint. A lost block is cheap; a distressed
  participant is not.
- Never leave a walking participant unattended in the CAVE!

> **Harness: decide and record.** The rig has a full-body harness on a retracting ceiling
> lifeline, guided by the oval rail 2.4 m up. It is standard in the obstacle-traversal studies
> on this platform. PACE walks in place at 1.0 m/s on a level base, a much lower fall risk, so
> harness use is a judgement call. Whatever you decide must be a fixed policy, applied
> identically to every participant and written into the ethics protocol and the methods. Do
> not improvise it per session.

---

## 1. Session design and scheduling

Each participant comes **twice**, for visits of about **80 minutes**.

Per visit: **1 standing block + 4 walking blocks** (~12 min each), one standing and one
walking calibration, plus baselines and breaks. Across both visits: **2 standing + 8 walking
blocks** per participant, roughly **1440 walking trials** and 360 standing.

> **Why 4 walking and 1 standing block per visit.** The confirmatory test is powered on
> walking trials. 1440 of them give power **0.93** at N = 12 and **0.96** at N = 15. The
> standing-versus-walking threshold comparison is within-participant, which cancels the
> between-participant spread. **One** standing block per visit therefore already gives power
> 0.994 (1.10× ratio). See [`Analysis/davidson_power/README.md`](Analysis/davidson_power/README.md).

**Watch for walking fatigue.** Four walking blocks mean ~48 min on the treadmill per visit.
Take the **5 min** break after every second walking block, and do not let a participant skip
it to finish early. QC flags easy-trial accuracy per block and compares the first third with
the last third. A flagged block means a tired participant, not necessarily bad data, but
record it in your session notes.

**Schedule the second visit 2–21 days after the first.** Shorter allows too little recovery;
longer makes the two visits hard to pool. `report.py` flags participants outside this window.

### Condition order: counterbalanced, flipped between visits

The order comes from a fixed allocation list built into the software. **Visit 2 runs the
opposite order to visit 1.** This crossover is deliberate. Every participant does each
condition once early and once late in a session, so tiredness cannot masquerade as a
difference between standing and walking.

**You do not compute anything.** The text at the top of the setup screen guides you in two steps:

1. **Type the participant number.** The screen shows the group (S = standing first in visit 1,
   W = walking first in visit 1) and the order of both visits, and asks for the visit.
2. **Type the visit number.** The screen lists every step of this visit, with the Block and
   Trial to type and the button to press:

```
P07   |   VISIT 2   |   Group W     (counterbalance slot 7)
Group W: visit 1 walking first, visit 2 standing first.     This visit: STANDING FIRST (crossed over)

Steps for visit 2: type Block and Trial as listed, then press the button in [ ]
1.    Block 0    Trial 2        Calibrate STANDING    [Start Calibration Standing]
2.    Block 0    Trial 2        Standing block 2 of 2    [Start Experiment]
3.    Block 1    Trial 2        Calibrate WALKING    [Start Calibration Walking]
4.    Block 1    Trial 5        Walking block 5 of 8    [Start Experiment]
...
```

Once Block and Trial are typed, the matching step and the file it will write are highlighted.
Values that match no step show a red warning. This is expected only when re-running a block
under a new Trial number (§7).

Two rules:

- **Always set the Visit field.** It is empty at launch, and Start and Calibrate refuse to run
  without it. It stays filled between runs for the same participant. If you type a different
  participant, type the visit again.
- **A replacement for a dropout takes the dropout's Counterbalance Slot.** This keeps the
  standing-first / walking-first split even.

Each condition is calibrated **once per visit**, and its blocks run back-to-back. The
threshold carries across them, so a break between blocks needs no recalibration.

---

## 2. Before the participant arrives: rig checklist

Work on the **head node**, in the project checkout, from a **graphical terminal**. A plain SSH
session gives `Can't open display :0`. The rig scripts read machine names and paths from
`rig/rig.env` (§12).

1. **Deploy the build**, only if it changed since last time.
   ```bash
   rig/ill.py stop     # stop render-node apps first, or their files stay LOCKED
   rig/dist.sh         # wipes each node, then recopies the Windows build
   ```
   `dist.sh` wipes before copying. **Never** fall back to a plain `scp` loop. Stale files on
   the render nodes make the walls flicker or split.

2. **Start the treadmill server** (needed for walking). The Modbus bridge `runTreadmillServer`
   belongs to the lab installation, not to this repository. It listens on UDP port 1338.
   ```bash
   ps aux | grep -i tread | grep -v grep    # note and kill a stale PID if present
   /path/to/runTreadmillServer
   ```

3. **Launch the experiment.**
   ```bash
   rig/start.sh        # stops old render apps, launches render nodes + head node
   ```

4. **Motion capture.** Warm the cameras up ~20–30 min beforehand. Recalibrate in Motive if a
   camera was bumped, the room temperature changed a lot, or the pose jitters. Head rigid
   body **Streaming ID = 1**.

5. **Check the rig before the participant is in it.**
   - Every CAVE wall shows the scene, with no black or split walls. Otherwise redeploy (§11).
   - The head rigid body is tracked. Nod and tilt the marker and check that the pose follows
     in all axes, not yaw only.
   - Both response buttons register.
   - The treadmill responds and the emergency stop works.

6. **Prepare the paperwork**: consent form, demographics sheet and the participant number.

---

## 3. Welcoming and briefing the participant

Brief every participant identically. Differences in wording become differences in the data.

1. Consent and paperwork first.
2. Explain the task and demonstrate the response buttons.
3. Fit the head marker firmly so it cannot slip during walking.
4. Let them walk in place on the treadmill for a minute at 1.0 m/s so their gait settles.
5. Run the **calibration** first. It doubles as task practice.

### Briefing script

Read the version in the participant's language, verbatim. Paraphrasing adds
between-participant variance.

**🇩🇪 Deutsch**

> „Gleich sehen Sie vor sich kurz ein kleines gestreiftes Feld aufblitzen. Es ist jedes Mal
> leicht nach links oder nach rechts geneigt. Ihre Aufgabe ist es, die Taste für die Seite zu
> drücken, zu der es geneigt war: linke Taste für links, rechte Taste für rechts.
>
> Die Neigung ist oft sehr klein, deshalb wird es sich schwierig anfühlen. Das ist so
> beabsichtigt. Wenn Sie unsicher sind, **raten Sie**. Geben Sie bitte immer eine Antwort, und
> antworten Sie eher **schnell** als lange überlegend.
>
> Es gibt zwei Arten von Blöcken. In manchen stehen Sie einfach still. In anderen gehen Sie auf
> der Stelle auf dem Laufband. Gehen Sie dabei in einem natürlichen, gleichmäßigen Tempo und
> schauen Sie geradeaus. Bitte schauen Sie nicht nach unten auf Ihre Füße.
>
> Jeder Block beginnt mit etwa einer halben Minute ruhigem Stehen. Entspannen Sie sich einfach
> und warten Sie, bis die Felder anfangen.
>
> Sagen Sie mir sofort Bescheid, wenn Ihnen schwindelig oder unwohl wird oder Sie sich unsicher
> fühlen. Wir können jederzeit abbrechen. Haben Sie noch Fragen?“

**🇬🇧 English**

> "You will see a small striped patch flash briefly in front of you. Each time, it will be
> tilted slightly to the left or to the right. Your job is to press the button for the side it
> leaned toward: left button for left, right button for right.
>
> The tilt is often very small, so it will feel difficult. That is intended. If you are not
> sure, **guess**. Please always give an answer, and answer **quickly** rather than
> deliberating.
>
> There are two kinds of blocks. In some you simply stand still. In others you walk in place
> on the treadmill. While walking, keep a natural, steady pace and look straight ahead. Do not
> look down at your feet.
>
> Each block starts with about half a minute of quiet standing. Just relax and wait until the
> patches begin.
>
> Tell me immediately if you feel dizzy, unwell or unsteady at any point. We can stop at once.
> Do you have any questions?"

Do **not** mention the stride cycle, the gait phase or the hypothesis. Participants who know
the timing hypothesis may change how they walk.

---

## 4. The setup screen: every field

Head node, `00_SetupScene`, the `ExperimentConductor` panel.

| Field / control | What to enter | Notes |
|---|---|---|
| **Participant** | participant number (integer) | Type it first. The schedule text updates live. |
| **Visit** | `1` or `2` | **Drives the crossover and is part of every file name, so it must be right.** Empty at launch; Start/Calibrate refuse to run without it. |
| **Counterbalance Slot** | *(not on the screen in the current build)* | The slot is the participant number. Needed only to re-fill an excluded participant's slot. |
| **Block** | `0` = standing, `1` = walking | The condition selector, nothing else. |
| **Trial** | the run number for this block | See [§6](#6-run-procedure-exact-values-to-type). Never reuse a (Block, Trial) pair. |
| **Speed** | `1.0` | Treadmill / scene speed in m/s. Leave at 1.0. |
| **Threshold** | *leave blank* | Manual override only. See [§10](#10-restarting-without-recalibrating). |
| **Threshold Standing / Walking** | *leave blank* | Pre-filled from the stored calibrations. |
| **Debug toggle** | **OFF** | Disables block logging. Unticked whenever the setup screen opens; tick it only on purpose for a test run. A red warning shows while it is on. |
| **Calibrate Standing** | starts the standing calibration | Sets condition = standing. |
| **Calibrate Walking** | starts the walking calibration | Sets condition = walking. |
| **Start** | starts the experimental block | Uses the current Participant / Block / Trial. |
| **Schedule text** (top of the screen) | *read-only* | Group, this visit's steps with Block/Trial, the step the fields match and the file it writes. |

**Controls during a run:** **Space** or **PageDown** starts the block after the baseline
notice. The participant responds with the Left/Right buttons. Do not touch the
projection-tuning keys (README §7) during a session.

---

## 5. Calibration (QUEST+)

Run each condition's calibration **immediately before its blocks**.

Set **Participant**, **Visit**, **Block** (`0` or `1`, matching the condition) and **Trial =
the visit number** (`1` in visit 1, `2` in visit 2). Then press **Calibrate Standing** or
**Calibrate Walking**, whichever the order label names first.

What happens:

- Five easy warm-up trials, then an adaptive run of roughly 40–60 trials with two interleaved
  trackers and occasional easy "catch" trials.
- At the end, the head-node overlay shows a green **DONE** or a red **UNRELIABLE** banner.
- **UNRELIABLE** means the run did not converge, catch trials were failed, or the estimate hit
  a limit. **Re-instruct the participant and run the calibration again.** Do not start blocks
  on an unreliable threshold.
- The threshold is stored for that condition and used automatically by its blocks. You never
  type it in.
- A file `pXX-vV-bY-tZ_calibration_<condition>.csv` is written. Every row also records
  participant, visit, counterbalance slot, block, trial and condition.

The standing calibration keeps the treadmill stopped; the walking calibration runs it. Each
threshold is thus measured in the state in which it will be used.

---

## 6. Run procedure: exact values to type

Run the first condition's **calibration, then its blocks**. Then switch once to the other
condition and do the same. Copy the Trial numbers from the tables.

> **The invariant:** never reuse a **(Block, Trial)** pair for one participant. Experimental
> blocks of a condition take Trial `1, 2, 3, …`, counting across both visits. Each visit's
> calibration takes Trial = the visit number. The tables below follow this rule.

### Worked example: a participant whose visit 1 is STANDING first

Read the order off the setup screen. Do not infer it from the participant number.

**Visit 1:** standing group first

| # | Action | Participant | Visit | Block | Trial | File written |
|---|---|---|---|---|---|---|
| 1 | Calibrate Standing | 14 | 1 | 0 | 1 | `p14-v1-b0-t01_calibration_standing.csv` |
| 2 | Start: standing block 1 | 14 | 1 | 0 | 1 | `p14-v1-b0-t01_standing.csv.gz` |
| 3 | Calibrate Walking | 14 | 1 | 1 | 1 | `p14-v1-b1-t01_calibration_walking.csv` |
| 4 | Start: walking block 1 | 14 | 1 | 1 | 1 | `p14-v1-b1-t01_walking.csv.gz` |
| 5 | Start: walking block 2 | 14 | 1 | 1 | 2 | `p14-v1-b1-t02_walking.csv.gz` |
| — | **5 min break** | | | | | |
| 6 | Start: walking block 3 | 14 | 1 | 1 | 3 | `p14-v1-b1-t03_walking.csv.gz` |
| 7 | Start: walking block 4 | 14 | 1 | 1 | 4 | `p14-v1-b1-t04_walking.csv.gz` |

**Visit 2:** **walking group first** (the order crosses over). Set **Visit = 2**.

| # | Action | Participant | Visit | Block | Trial | File written |
|---|---|---|---|---|---|---|
| 1 | Calibrate Walking | 14 | 2 | 1 | 2 | `p14-v2-b1-t02_calibration_walking.csv` |
| 2 | Start: walking block 5 | 14 | 2 | 1 | 5 | `p14-v2-b1-t05_walking.csv.gz` |
| 3 | Start: walking block 6 | 14 | 2 | 1 | 6 | `p14-v2-b1-t06_walking.csv.gz` |
| — | **5 min break** | | | | | |
| 4 | Start: walking block 7 | 14 | 2 | 1 | 7 | `p14-v2-b1-t07_walking.csv.gz` |
| 5 | Start: walking block 8 | 14 | 2 | 1 | 8 | `p14-v2-b1-t08_walking.csv.gz` |
| 6 | Calibrate Standing | 14 | 2 | 0 | 2 | `p14-v2-b0-t02_calibration_standing.csv` |
| 7 | Start: standing block 2 | 14 | 2 | 0 | 2 | `p14-v2-b0-t02_standing.csv.gz` |

### The mirrored case

For a participant whose visit 1 is **WALKING first**, use the same tables with the condition
groups swapped: walking first in visit 1, standing first in visit 2. The numbering rule is
unchanged. Walking blocks take Trial `1…8` across the two visits, standing blocks `1…2`, and
each visit's calibrations take the visit number.

---

## 7. During a block, and what to do when something goes wrong

**Normal course of a block.** After you press Start, a "stand still and relax" message is held
for 30 s with the belt stopped. The participant then presses Space/PageDown, and the block
begins with five easy training trials. Near the end of a walking block, an on-screen message
warns that the treadmill will stop. The belt then ramps down and a 30 s cool-down is held. Let
both baselines run in full, as they are matched by design.

**Watch the head-node overlay** during the block: running accuracy (~75 % near threshold is
normal), progress and tracking status.

| Situation | What to do |
|---|---|
| Participant reports dizziness or discomfort | Stop the block immediately. Belt down, lights up, offer a seat and water. Decide together whether to continue, and note it. |
| Accuracy sits near 50 % | They may have misunderstood the mapping or be guessing throughout. Stop, re-instruct, re-run the calibration. |
| Accuracy sits near 100 % | The threshold is too easy, usually from a bad calibration. Re-run the calibration. |
| Almost all responses on one side | Check the response device for a stuck button **before** blaming the participant. |
| Tracking drops out mid-block | Stop, fix the marker or the Motive calibration, and re-run that block under the **next unused Trial number**. |
| Block aborted part-way | Do not reuse its Trial number. Re-run under the next free number and note which file is the partial one. |
| Participant needs a longer break | Fine between blocks. No recalibration is needed within a condition. |

**Never re-run a block under a Trial number you have already used.** Nothing is overwritten.
The rig writes the second log as `…-1.csv.gz`, and the pipeline keeps it as a separate block
(`run-104` for a reused Trial 4). But which file is which is then recorded only in your notes.

**Keep partial logs.** An aborted or disconnected block still holds usable trials, because the
pipeline reads every frame up to the break. Copy it off the rig with the others. If Unity
crashed, the block was never saved under its proper name. Its frames are in the logger's
temporary file, whose path the Unity log prints as `Logging to: …`. Copy that file, rename it
to the block's intended name (e.g. `p07-v1-b1-t03_walking.csv`), and note it in the session log.

---

## 8. After each visit: collecting the data

The head node writes loose logs to `ExperimentConductor.dataStoragePath`, or to the player's
persistent-data folder if that field is empty: `pXX-vV-bY-tZ_<condition>.csv.gz` for blocks
and `pXX-vV-bY-tZ_calibration_<condition>.csv` for calibrations. `V` is the visit; the
pipeline also reads logs without the `-vV` part.

1. **First, back up the raw logs off the rig.** Copy the whole set somewhere safe. They cannot
   be regenerated.
2. Copy the loose CSVs into **`data/`** on the analysis machine (`config.RAW_DIR`).
3. Convert and run the pipeline (§9). This is safe after each visit; visit 2 simply adds more
   `run-` recordings.
4. Log the participant number, visit, date, the condition order shown on the setup screen,
   any aborted blocks, and anything unusual.

---

## 9. Converting and checking the data

From the repo root:

```bash
python -m Analysis.bidsify        # loose CSVs -> BIDS under data/, raw archived to sourcedata/
python -m Analysis.run_pipeline   # QC, gait, stride phase, GLMM, prevalence, report
```

**How multi-block data is stored.** Each block becomes its own BIDS recording, identified by a
`run-` entity taken from the **Trial** field:

```
data/sub-12/ses-02/motion/
  sub-12_ses-02_task-ILL_acq-standing_tracksys-optitrack_run-01_motion.tsv
  ... run-02                                           (2 standing blocks)
  sub-12_ses-02_task-ILL_acq-walking_tracksys-optitrack_run-01_motion.tsv
  ... run-02 … run-08                                  (8 walking blocks)
data/sub-12/ses-01/beh/
  sub-12_ses-01_task-ILL_acq-standing_run-01_beh.tsv   (calibrations, run-01 / run-02 per visit)
  sub-12_ses-01_task-ILL_acq-walking_run-01_beh.tsv
```

Both visits fold into `ses-02` and are distinguished by `run-`. The analysis pools all of a
participant's walking runs, so each participant counts once, with all of their trials.

**Check after conversion:**

- **File count.** `data/sub-XX/ses-02/motion/` should hold one `_run-*_motion.tsv` per block
  you ran: 8 walking + 2 standing after both visits. An extra `run-1NN` file (e.g. `run-104`)
  means a Trial number was reused. `bidsify` prints a warning naming both logs; use your
  session notes to tell which block is which. If a run is missing, check that its raw log
  was copied into `data/`.
- **Counterbalance.** In `data/derivatives/summary/counterbalance_qc.tsv`, every participant
  should show two visits, `crossed_over = True` and an inter-visit gap of 2–21 days. A failure
  means the Visit field was set wrongly.
- **Report.** `data/derivatives/summary/report.html` has one row per block, plus the population
  results and the standing-versus-walking comparison.
- **Data quality.** `data/derivatives/summary/data_quality.md` counts the blocks and trials
  recorded, kept and excluded per participant and condition, each exclusion with a standard
  label. A manual exclusion is labelled in `config.EXCLUSION_LABELS`.
- **Exclusions.** Check the report's inclusion/exclusion table for blocks flagged for unusable
  gait coverage, chance-level or reversed answers, or an unreliable calibration. Re-run a
  calibration flagged on the rig before its blocks. Otherwise its blocks stay in the analysis
  but are harder or easier than planned.

---

## 10. Restarting without recalibrating

To resume a participant on a **saved** threshold, type it into the **Threshold** field (or the
per-condition **Threshold Standing / Walking** fields) before pressing Start. Easy and hard
levels are derived from it automatically. Take the value from the `# threshold_deg=…` header
line of that participant's earlier calibration CSV. Leave the field blank to use the most
recent calibration.

Use this only to recover from a crash or an interrupted session. It is not a way to skip
calibration in a fresh visit, since each visit calibrates each condition once by design.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| CAVE walls flicker, split, or distort progressively | Stale files on the render nodes. Redeploy with `rig/ill.py stop`, then `rig/dist.sh` (which wipes first). Never plain `scp`. |
| Setup scene: top monitors black, text only on the bottom | Read the `SetupDisplayActivator` log. If it reports 1 display, the two monitors form one spanned display. This is a window-coverage issue, not an inactive display. |
| `Can't open display :0` | Launch `rig/start.sh` from a graphical terminal on the head node, not over plain SSH. |
| `runTreadmillServer` aborts, `invalid pointer` | A stale instance holds port 1338. Kill the old PID and relaunch. |
| Head pose jitters or jumps by metres | Fix OptiTrack at the source: calibration, markers, rigid-body definition. `clean.py` only removes residual spikes. |
| Head orientation logs yaw only | Set `headRigidBodyId` to the head's Motive User ID. Check that `head-mocap-rot-x/z` move when nodding and tilting. |
| Strong response side bias | Check the response device for a stuck button first. |
| Calibration banner **UNRELIABLE** | Re-instruct, re-run the calibration. Do not proceed. |
| An extra `run-1NN` recording after conversion | A Trial number was reused. Both logs are kept; the second is stored as `run-1NN` with a warning. Identify the block from your session notes. |
| A block is missing after conversion | Its raw log was not copied into `data/`. Copy it from the backup (§8) and re-run `bidsify`. |
| `counterbalance_qc.tsv` says the order did not cross over | The Visit field was wrong in one visit. Record it; the confound cannot be repaired after the fact. |

---

## 12. One-time rig setup

The repository has **placeholders** wherever the original installation had a machine name, an
address or a path. Replace them for your rig, then build (`rig/build.sh`, and
`rig/build.sh windows` for the render nodes) and deploy (`rig/deploy.ps1` or `rig/dist.sh`).

**Rig scripts.** Copy `rig/rig.env.example` to `rig/rig.env` (git-ignored). Fill in the head
node, its checkout path, the render nodes as `host:user` pairs, the SSH key, the Unity editor
path and the build folder on the render nodes.

**Signing key.** The render nodes run a small listener (part of the lab installation) that
starts and stops the player on Ed25519-signed commands from `rig/ill.py`. Put its 64-hex seed
in the file named by `ILL_SIGNING_KEY_FILE` (`chmod 600`) or export `ILL_SIGNING_KEY`. It
never goes into the repository.

**Unity scenes and prefab** (set the Inspector values, then rebuild):

| Where | Field | Placeholder | Set to |
|---|---|---|---|
| `ILL-UniCave 1 1.prefab`, `MasterTrackingData` | `_HeadNodeMachineName` | `HEAD-NODE` | the head node's machine name |
| `ILL-UniCave 1 1.prefab`, each display | `MachineName` | `RENDER-NODE-1/2/3`, `HEAD-NODE` | the machine that drives that wall |
| `ILL-UniCave 1 1.prefab`, `VRPNTrack` objects | `trackerAddress` | `Head@tracking-pc` | `<rigid body>@<tracking PC>` |
| `ILL-UniCave 1 1.prefab`, OptiTrack client | `ServerAddress` | `tracking-pc` | the Motive PC |
| `00_SetupScene`, `02_Calibration`, prefab: `NetworkingSync` | `headNodeIP` | `127.0.0.1` | the head node's address |
| `00_SetupScene`, `ExperimentConductor` | `dataStoragePath` | *(empty)* | where the head node writes logs; empty = the player's persistent-data folder |

Each computer compares these names with its own (`System.Environment.MachineName`). A display
whose `MachineName` matches no computer stays black (the comparison ignores case).
`_HeadNodeMachineName` must match the head node's name exactly, including case, or no computer
takes the server role.

**Scene wiring** (already done in the committed scenes). The walking calibration scene needs a
`TreadmillController`. The setup scene's **Calibrate Standing** / **Calibrate Walking**
buttons, the **Calibration Order Label**, the **Visit** field and the per-condition threshold
fields are assigned to their Inspector slots. Leave the buttons' `OnClick` lists empty,
because the handlers are wired in code. The setup screen has no **Counterbalance Slot** field;
the slot is the participant number.

---

## 13. Reference card

```
Field    Meaning                    Values
-------  -------------------------  ------------------------------------------------
P        participant number         1, 2, 3, …
Visit    which visit                1 or 2      (visit 2 = order crossed over; no start without it)
Slot     counterbalance slot        = P (no field in the current build)
Block    condition selector         0 = standing, 1 = walking
Trial    run index                  blocks:       1,2,3,… per condition, across visits
                                    calibration:  the visit number (1 or 2)
Speed    treadmill / scene speed    1.0 m/s
Debug    logging switch             OFF for every real run

Per visit:  1 standing block + 4 walking blocks + 1 standing cal + 1 walking cal
Per person: 2 standing + 8 walking blocks over two visits, 2–21 days apart
Block:      ~12 min, ~180 trials, 30 s baseline before and after
Breaks:     5 min after every SECOND walking block (48 min walking per visit)
```
