# combined corpus dataset

- Row count: 26 (one row per book in the st1 table)
- Join key: `book`
- Books with both st1 and nd1 results: 26
- Books with st1 only (nd1 pending): 0

Every st1 column is prefixed `st1_`, every nd1 column `nd1_` (join key `book` is unprefixed). Rows for books without an nd1 sidecar yet have blank `nd1_*` columns — this is expected (nd1 is a partial, in-progress run) and shows current benchmark scope at a glance, not missing data to chase down.

