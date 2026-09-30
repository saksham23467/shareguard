# ShareGuard: brutal technical review and redesign

Prepared for Saksham Bansal, 30 Sep 2026. This builds on `shareguard_platform_spec.md` and the room's points about blur-first safety metrics, line packing, layout rules, training on OCR-noised text, self-capture and test-set sizing.

**Rule for this document:** a component survives only if it moves a number in the benchmark table in §5.

---

## 1. Original architecture (v1)

```
Capture → tile diff → [pre-blur dirty tiles] → EasyOCR detector (NPU) → EasyOCR reader (NPU, one call per line)
       → regex/checksums (CPU) + DistilBERT PII tagger (NPU) + face detector (NPU) → fuse → blur (GPU) → shared window
Optional: Qwen3-0.6B rule authoring, Whisper audio redaction, virtual camera
```

## 2. Problems: the ten questions, answered honestly

1. **What's actually novel?** None of the *models*. They're off-the-shelf AI Hub models. The novel part is the **streaming safety system**: fail-closed redaction, where a region is shown only once it's been verified, plus incremental OCR over a changing screen. If the pitch leads with models, it's a wrapper. If it leads with the verified-unblur loop and its guarantees, it isn't.
2. **What genuinely needs on-device AI?** All of the per-frame path. Sending your screen to a cloud service in order to hide it from viewers defeats the purpose, and a round trip adds more latency than a frame budget allows. That argument is strong and honest.
3. **What genuinely benefits from the NPU?** Less than v1 implies.
   - A static screen needs no inference at all, and typing changes one line.
   - **The NPU matters in bursts**, like a window switch, tab change or fast scroll, when dozens of lines need reading at once and *time until safe text reappears* depends directly on reader throughput. It also matters for **sustained power and thermals** over a one-hour call.
   - Those two claims have to be proven with numbers, not asserted.
4. **What would a judge dismiss as an API wrapper?** "EasyOCR + regex + Gaussian blur." That *is* v1's core, minus the scheduling.
5. **What could a cloud API replace?** Nothing in the hot path, legitimately. Qwen rule authoring could be replaced by any cloud LLM, which is exactly why it's weak: **cut it**.
6. **What could an ordinary CPU replace?**
   - Diffing, rules, tracking and fusion belong on the CPU anyway.
   - **Face detection** at 0.45 ms on the NPU would also be fine on the CPU, so don't sell it as an NPU win.
   - The async PII tagger could probably keep up on the CPU at low text rates.
   - **Windows.Media.Ocr**, the built-in CPU OCR, is the baseline a judge will name. It has to be in our benchmark, or we look like we're hiding it.
7. **Technically impressive but unnecessary:** the local LLM, VLMs, embeddings and retrieval, audio redaction and the virtual camera.
   - **Embeddings and retrieval:** I looked for a measurable use and there isn't one worth its cost. For example, "retrieve similar past secrets" does nothing that exact hashing of already-read lines doesn't do better.
   - **Rejected on purpose.** Say so in the docs; judges respect a documented "we considered X and here's the number that ruled it out".
8. **What would show real ML engineering?** Adapting the reader to the domain (screen fonts, random key strings), per-layer mixed-precision quantization, **calibrated confidence and a statistically controlled unblur rule**, a precision cascade, and training the tagger on OCR-noised text.
9. **What would show systems engineering?** Incremental OCR with scroll compensation, deadline-aware priority scheduling across NPU, GPU and CPU, line packing, a zero-copy GPU path, a fail-closed state machine per region, EP-context start-up, thermal behaviour, and a degradation ladder.
10. **What can we benchmark?** Everything in §5. Every row is a with-vs-without comparison.

**The core problem with v1:** its safety is a design assumption ("blur first"), not a *measured, controlled* property. Its speed claim is a per-model latency, not a user-facing metric. v2 fixes both.

---

## 3. Improved architecture (v2)

```
                          ┌─────────────── Text source: UI Automation (CPU, free) ──────────────┐
Capture (GPU texture) ─► Tile hash + scroll/motion estimate (CPU) ─► Region state machine (CPU)
                                                         │             states: UNVERIFIED(blur) → VERIFIED_SAFE(show)
                                                         │                     → SENSITIVE(blur) ; any change → UNVERIFIED
                                                         ▼
                      Deadline-aware scheduler (CPU)  ── priority = focus/cursor proximity × area × age
                                                         │
          ┌──────────────────────────────────────────────┼──────────────────────────────┐
          ▼                                              ▼                              ▼
  Text detector w8a8 (NPU)                    Face detector (CPU or NPU)       [idle: nothing runs]
          │
  Line cache (hash of line pixels → verdict)  ── hit → verdict reused, zero inference
          │ miss
  Line packer: short lines packed into 64×800 strips; long lines alone (CPU)
          ▼
  Reader stage 1: domain-tuned CRNN, mixed precision (NPU)  → text + per-character confidence
          │
  Calibrated risk score (CPU): P(line contains a secret the rules would miss | OCR output, confidence, text shape)
          │── low risk  ─────────────────────────────────────────────────────► VERIFIED_SAFE
          │── pattern hit (checksum/regex/layout rule) ─────────────────────► SENSITIVE
          │── uncertain ─► Reader stage 2: higher-precision reader on that line only (NPU fp16/w8a16, or GPU)
          │                  └─ still uncertain ─► stays blurred (fail-closed), shown as "unverified" in HUD
          ▼
  PII tagger, trained on OCR-noised text (NPU, async, free text only; form fields use layout rules on CPU)
          ▼
  Fusion + tracker (CPU) ─► Blur composite (GPU, zero-copy) ─► Output window (capture excluded from own input)
```

---

## 4. Exact technical improvements (each with its purpose)

| # | Improvement | Type | Purpose (the number it moves) |
|---|---|---|---|
| I1 | **Region state machine with verified unblur.** Nothing is shown until it's been read and cleared. Any pixel change resets it to blurred. | Systems + safety | Leak time ≈ 0 by construction; turns safety into the **miss rate** (data scientist) |
| I2 | **Calibrated risk score + controlled threshold.** Per-line features: CTC character confidences (min, mean), randomness of the string, character-class mix, length, near-miss distance to each pattern (edit distance after confusable mapping), layout-label context. Fit logistic regression or small gradient-boosted trees on OCR-noised synthetic screens; calibrate with isotonic or temperature scaling. Choose the unblur threshold on a calibration split so the miss rate stays under a target α, with a finite-sample bound (risk-control or conformal-style), and report it on a separate test split. | ML | Miss rate vs over-blur curve (data scientist's curve), with a stated bound instead of a hand-set threshold |
| I3 | **Precision cascade (early exit).** Fast 8-bit reader for every line; re-read only *uncertain* lines with a higher-precision reader (fp16/w8a16 on NPU, or GPU). | ML + systems | Keeps 8-bit speed on ~most lines and near-float accuracy where it matters. Measure % escalated, character error rate, unblur time |
| I4 | **Domain-adapted reader.** Fine-tune the EasyOCR CRNN on synthetic screen crops: UI fonts, monospace terminals, syntax-highlighted code, dark mode, 100/125/150% scaling, random key strings, Indian names, IDs. Then quantize with per-layer sensitivity analysis, keeping sensitive layers at 16-bit activations (quantization-aware fine-tuning if post-training quantization isn't enough). | ML | Character error rate on key crops and small dark text; *the* fix for `l/1/I`, `O/0` |
| I5 | **Line packing** (deep learning engineer): several short crops per 64×800 strip, split by CTC column position. Validated against single-line reading. | Systems + ML | NPU calls per window switch, unblur time; character error rate unchanged |
| I6 | **Incremental OCR:** line-level pixel-hash cache plus scroll compensation (row-hash matching shifts cached verdicts). | Systems | Inference calls per second during scrolling; unblur time for scrolls ≈ 0 for already-seen content |
| I7 | **Deadline-aware scheduler:** priority to regions near the cursor or focused control and to large regions; NPU queue ordering; stale-work cancellation when a region changes again before being read. | Systems | p50/p95 *time until the region the viewer is looking at is readable*, and wasted inference on superseded frames |
| I8 | **UI Automation as a free text source:** for native apps exposing text, cross-check it against detector boxes so those lines skip recognition. Pixel-only content (images, video, remote desktop, VMs, terminals in VMs) still goes through the NPU. | Systems | NPU calls saved per window switch; answers "why not just read Windows' text?" *(Experiment: UI Automation coverage and box alignment vary by app; keep only if it measurably helps.)* |
| I9 | **Layout rules** (nlp engineer): label-to-value linking ("Name:", "Account No") on CPU. Tagger reserved for free text and trained on OCR-noised text. | ML | Recall on form fields vs tagger-only; tagger recall on Indian names reported separately |
| I10 | **Graceful degradation ladder:** NPU → GPU → CPU backends, chosen at start-up. Under load, the system gets *more* conservative: it stays blurred, it never skips checks. HUD shows the mode. | Systems | Miss rate stays constant across backends; only unblur time degrades (that's the proof the NPU buys usability, not safety) |
| I11 | **Start-up and thermals:** pre-compiled EP-context binaries; `htp_performance_mode` switched between burst (on change) and power-saver (idle) [SW-DEP]. | Systems | Cold start time; sustained p95 over 15 minutes; battery drain |
| I12 | **Self-capture safety** (machine learning engineer): capture a single app window or a second display; test the display-affinity flag in Teams and Meet on day 1. | Systems | Pass/fail matrix per meeting app |

**Cut from v1, with reasons:**
- **Qwen rule authoring:** replaceable by the cloud, and moves no metric.
- **Audio redaction:** a separate product, so it dilutes the demo.
- **Virtual camera:** stretch goal only (it needs a C++ media source).
- **VLMs, embeddings and retrieval:** no metric they'd improve.
- **Face detection:** stays for the camera feed only, and isn't claimed as an NPU win.

---

## 5. Benchmarks proving each improvement

Test data is the nlp engineer's synthetic screen generator. The data scientist owns the scoring script and the bootstrap, with the screen as the resampling unit.
- **Headline safety claim:** 3,000+ planted secrets.
- **Per condition:** "a few percent" bounds for each condition group.
- **Worst cases:** about 3,000 secrets for each of the 3–4 hardest groups (for example small dark-mode terminal text at 150%). This is the amazon applied scientist's correction.

| Improvement | Metric | Comparison | Expected shape of the win |
|---|---|---|---|
| I1 | Leak time (p99), miss rate | Blur-after-OCR vs blur-first | Leak time falls from OCR latency to ~a frame. The miss rate becomes the number that matters. |
| I2 | Miss rate at a fixed over-blur; bound on miss rate | Hand threshold vs calibrated + controlled threshold | Lower misses at equal over-blur, plus a guarantee on the synthetic distribution |
| I3 | Character error rate, % lines escalated, unblur p95 | All-8-bit / all-float / cascade | Close to float's character error rate at close to 8-bit's speed |
| I4 | Character error rate on keys, small dark text, Indian names | Stock reader vs adapted; per-layer precision variants | Largest accuracy jump in the project |
| I5 | NPU calls per window switch, unblur p95, character error rate | One line per call vs packed | Fewer calls, character error rate unchanged, or packing limited to short lines |
| I6 | Inference calls/s while scrolling; unblur time on scroll | No cache vs cache + scroll compensation | Near-zero work for content already read |
| I7 | Time until the viewer-focused region is readable; wasted inference | FIFO vs priority + cancellation | Focused region readable much sooner after a switch |
| I8 | NPU calls saved; agreement rate with OCR | With vs without UI Automation | Keep only if the savings are real |
| I9 | Form-field recall; free-text recall; Indian-name recall | Tagger only vs layout rules + tagger | Better recall on forms; honest name-specific number |
| I10 | Miss rate and unblur p50/p95 **per backend** | NPU vs GPU vs CPU vs Windows.Media.Ocr baseline | **Identical miss rate, much slower unblur on CPU.** This is the central Snapdragon slide. |
| I11 | Cold start; p95 over a 15-minute sustained replay; battery drain over a 30-minute replay (`powercfg /batteryreport`, fixed brightness) | With vs without EP cache; NPU vs CPU | Flat NPU line vs a CPU line that throttles; lower drain on the NPU. No NPU-power claim without a counter. |
| Offline | All of the above with networking off | Online vs offline | Identical |
| Cloud | End-to-end unblur vs a cloud OCR call on **synthetic screens only** | Local vs cloud | Cloud slower plus it's a privacy violation. The slide says the screens were synthetic. |

**Statistical rules:** paired comparisons on the same screens, bootstrap confidence intervals, a warm-up before timing, and per-condition reporting. Never report a single blended number.

---

## 6. What the live demo shows (3 minutes)

1. **Split view:** left is the presenter's real screen, right is *exactly what Meet viewers receive*. A HUD overlays region states: red means unverified, green means verified safe, orange means secret.
2. **Typing:** open a `.env` file and type a key. It stays blurred the entire time (orange); ordinary text turns green within a moment.
3. **Window switch** to a mock bank page with an Aadhaar and PAN: the screen briefly goes red, then green regions appear focus-first, and the IDs stay orange. A live counter shows time-to-readable.
4. **The Snapdragon moment:** flip the backend to CPU. Viewers still never see a secret (same miss rate), but the screen stays red for visibly longer on every switch, and the p95 chart climbs. Flip back to NPU.
5. **Hard case:** a small, dark-mode terminal key with confusable characters. The cascade escalates it (HUD flashes "re-read at high precision") and it's caught.
6. **Airplane mode on:** everything keeps working.
7. **Closing slide:** the miss-rate vs over-blur curve with its bound, the per-backend table, and the 30-minute battery comparison.

---

## Build order (one developer, ~14 days)
- **Days 1–2:** day-1 gates (NPU reader speed and character error rate, window-share and self-capture matrix, ARM64 dependencies), then I1, I6 and the rules.
- **Days 3–6:** I4 (synthetic data and reader fine-tune), I5, I3.
- **Days 7–9:** I2 calibration and threshold, I9, the tagger.
- **Days 10–11:** I7, I10, I11, and I8 if time allows.
- **Days 12–14:** benchmarks, installer, docs, demo video.

**If time runs short, cut in this order:** I8, then I7, then the tagger. Never cut I1, I2 or I4. They're the difference between a wrapper and a system.
