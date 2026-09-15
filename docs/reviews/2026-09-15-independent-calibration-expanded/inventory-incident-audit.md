# Candidate-inventory incident audit

Date: 2026-09-15. Read-only audit; no canonical inventory, manifest, waveform,
or numerical/LAL files were changed or downloaded.

## Initial incident evidence

The frozen selection record records inventory SHA256
`dfe5c6b1752c3f8adf5bc2a6910d4266de63d8ce41f6a7cb0d73bb54c8710e37`
(`selection-round1.json`, six selected simulations and 24 selected rows). The
current `candidate-inventory.json` and the preserved
`candidate-inventory-during-prepare.json` are byte-identical to each other, but
both hash `76e7832aa740b11d569a12f57ac871dd504a87be4c7cef3e390ac6c1c840417a`.
Neither therefore reproduces the inventory digest recorded by the selection.

The downstream manifest independently records the current inventory digest
(`76e7832…`) while selection records the earlier digest (`dfe5c6…`). This was a
real provenance mismatch even though the selected six IDs remained visible in
the selection record and the manifest contained 24 rows. The original full
inventory bytes are unavailable; no timestamp should be treated as evidence of
those bytes.

For the six fetched catalog records, the raw cache files exist and each
SHA256 matches its `raw_record_sha256` in the current inventory. This supports
record/cache integrity for those six records only; it cannot recover or
validate the missing pre-freeze inventory bytes, nor the 175 catalog-only
alternatives.

## Transparent replay and independently checked result

The repair used a **replay**, not recovery of the missing original inventory.
`selection-replay.json` records the current snapshot digest and explicitly
identifies the incident. Its 181-entry `candidate_order` and 24 case rows are
exactly equal to the original selection record. The replay manifest records
the replay selection hash and current inventory hash. All 72 source-file
identity tuples (key, MD5, size, and content URL) in the original prepared
manifest match the replacement inventory's records. The replay generated all
24 NPZs, including the q=1.2 case that later failed NR quality screening; each
of the 24 replay NPZs has array-for-array equality with its original prepared
NPZ.

This proves that the selected cases, source inventories, and replayed waveform
files are unchanged and that the current snapshot gives a coherent replay.
It does **not** prove that the unrecovered 181-entry original inventory bytes
were identical to the current snapshot.

## Final recommendation

1. Preserve the current inventory, during-prepare copy, replay selection, and
   replay manifest unchanged as incident evidence; do not relabel them as the
   original `dfe5c6…` bytes.
2. Keep the replay as the authoritative reproducible provenance for the 24
   selected rows, with its explicit incident note and current snapshot hash.
3. If a byte stream hashing exactly to `dfe5c6…` ever appears, retain it as a
   separately named recovered artifact with recovery source/time; do not merge
   it silently into the replay.
4. Continue to describe the full original inventory freeze as unrecovered, even
   though selection, source-file identities, and waveform arrays were verified
   unchanged by replay.

The incident does not invalidate the reported five NR-quality passes or the
q=1.2 extraction failures. Replay now provides reproducible linkage for the 24
selected rows; only the claim that the entire original 181-entry inventory
freeze was byte-identical remains unavailable.
