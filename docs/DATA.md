# Data interfaces

All commands accept user-supplied paths. Relative paths are resolved from the current working directory; paths inside an export manifest are resolved from that manifest's directory. Source data are never modified.

## Tokenizer

`train_windows.npy` is float32 `[N,64,D]` in physical feature units, with D=263 for HumanML3D or D=499 for Ω. `train_tmr.npy` is `[N,256]` from the frozen motion teacher for those exact windows. `feature_stats.npz` contains training-only `mean/std [D]`. Teacher embeddings must match the training windows.

Ω geometry loss additionally requires a SOMA77 template NPZ with `parents [77]` and `offsets [77,3]`. The template defines the skeleton used by the geometry loss.

For HumanML3D, use the model bundle's feature statistics, matching the standard
Guo evaluator normalization. The frozen motion teacher uses its own TMR
normalization: convert a window to physical Guo features, normalize with the
teacher's mean/std, encode it, and L2-normalize its latent mean. Save one teacher
vector for each exact training window. Ω uses the matched SOMA motion teacher.

## Text

Conditioning is frozen Flan-T5-XL token states `[L,2048]`. Append one sentence token with the frozen TMR **text** embedding multiplied by 8 in its first 256 channels and zeros elsewhere. This text encoder is distinct from the TMR **motion** teacher used to train the tokenizer. The joint text representation is left padded; motion latents are right padded. Every sample must have a valid text token.

`semoco.generator.pipeline.append_retrieval_token` assembles one unpadded sequence. `TextConditioner.encode(captions, retrieval_embeddings)` encodes new prompts and constructs a padded batch; it takes the matching TMR text embeddings.

## Export manifest

The manifest is a JSON list, for example:

```json
[
  {
    "motion": "clips/0001.npy",
    "text": "text/0001.npz",
    "reconstructed_joints": "recovered/0001.npz"
  }
]
```

Motion is an already selected physical `[4T,D]` crop. The text NPZ contains `text [L,2048]` including its sentence token. Omit `reconstructed_joints` to recover joints automatically from the codec's continuous reconstruction (using the packaged SOMA77 template on Ω). If an external Ω `reconstructed_joints` NPZ is supplied, it must contain:

- `normalized_features [4T,D]`: this exact codec's continuous reconstruction using real q0 and the pre-RVQ motion latent;
- `joints`: recovered physical `[4T,22,3]` for HumanML3D, or `[1+4T,77,3]` for Ω, including the initial anchor pose for Ω.

For Ω, obtain `normalized_features` with `encode_flow_targets` followed by `decode_continuous`. Denormalize and recover joints with the SOMA coordinate/skeleton runtime. Export verifies the reconstruction against the retained features. Auxiliary targets come from this frozen decode, not a different tokenizer or unrelated ground-truth trajectory.

Export writes:

```text
train.index.json                  code_start, code_len per aligned crop
train.codes.npy                   [total_tokens,1] q0
train.kin_latent.npy              [total_tokens,512] pre-RVQ motion latent
train.text_index.flantmr.json     text_start, text_len in the same row order
train.text_emb.flantmr.npy        [total_text_tokens,2048]
targets.npy                      [total_tokens,264 or 924]
kin_stats.npz                    training latent mean/std
targets_norm.npz                 training auxiliary mean/std
```

The generators require only q0 from the eight-code tokenizer. Seven motion codes remain part of tokenizer training and discrete reconstruction. They are not the Flow training target.

The export command operates on an explicit manifest in memory. For large corpora, export shards independently and concatenate arrays while adjusting offsets; compute statistics globally over training tokens, never by unweighted averaging of shard statistics. Existing aligned caches can be used directly in this layout.

## Generation input and output

Input NPZ: `text [B,L,2048]`, boolean `text_valid [B,L]`, and integer `frames [B]`. Optional `q0 [B,T]` reuses an already sampled semantic sequence; optional `noise [B,T,512]` fixes Flow's float32 initial noise.

HumanML3D uses `T=floor(frames/4)`; Ω uses `T=ceil(frames/4)`. HumanML3D evaluation supplies aligned multiples of four. The output `offsets` defines the actual unpadded lengths; a non-multiple HumanML3D request may produce fewer than `frames` records because it uses floor division. Ω output is cropped to the requested number of transition records.

The decoder processes each valid token span independently to prevent padded tokens from changing clip boundaries. Decoder outputs remain in feature space. Physical Guo263 and UMR499 recovery require the matching statistics, coordinate convention, skeleton, and for Ω the initial anchor.
