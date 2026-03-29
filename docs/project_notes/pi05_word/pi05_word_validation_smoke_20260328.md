# pi05_word Smoke Validation 2026-03-28

## Scope

This is a minimal Stage-A-style smoke validation for `pi05_word`, aligned with
`docs/project_notes/pi05_word/pi05_word_validation.md`.

It validates technical feasibility only:

1. `pi05_word` can load the original `pi05` checkpoint weights.
2. We can extract fused image+instruction hidden states without modifying the FM head.
3. A lightweight semantics head can consume the pooled hidden state and produce finite logits.
4. The original action pathway still runs.

It does **not** validate semantic accuracy yet.

## Validation script

Script:

```text
scripts/pi05_word_hidden_probe_smoke.py
```

Output JSON:

```text
outputs/pi05_word_validation/pi05_word_hidden_probe_smoke.json
```

## Run configuration

- device: `cuda`
- dtype: `bfloat16`
- checkpoint: `lerobot/pi05_libero_base`
- tokenizer: `google/paligemma-3b-pt-224`
- task prompt: `pick up the bowl`
- image input: synthetic random RGB tensor
- FM smoke test: `num_steps=1`

## Key results

- checkpoint weights loaded successfully
- remapped keys: `812`
- missing keys: none
- unexpected keys: none

Hidden-state extraction:

- prefix embeddings shape: `[1, 456, 2048]`
- fused prefix hidden shape: `[1, 456, 2048]`
- pooled hidden shape: `[1, 2048]`
- cached KV layers: `18`
- pooled hidden L2 norm: `32.60358428955078`

Lightweight semantics head:

- primitive logits shape: `[1, 10]`
- target logits shape: `[1, 6]`
- relation logits shape: `[1, 6]`
- all logits finite: yes

Original action path:

- action chunk shape: `[1, 50, 32]`
- action chunk finite: yes
- action chunk mean abs: `0.04334717243909836`

## Conclusion

This smoke validation supports the core Stage-A feasibility claim:

- `pi05_word` can reuse the original `pi05` weights.
- The fused image+instruction representation is directly accessible.
- A lightweight semantics head can be attached on top without touching the FM head.
- The copied policy still preserves the action-generation path.

## Important caveats

- The image was synthetic, not a real dataset frame.
- The semantics head was not trained; this only validates tensor flow and attachment feasibility.
- This is **not** a probing-quality result, so it does not say anything yet about slot accuracy,
  structural validity, or Level-1 vs Level-2 comparison.

## Recommended next step

Move to the next minimal research-valid step from `docs/project_notes/pi05_word/pi05_word_validation.md`:

1. replace the synthetic image with real image+instruction examples
2. save hidden states
3. train a small single-step probe head for a few slots such as:
   - primitive
   - target
   - relation
