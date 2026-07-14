"""Genre-view table for the masters report. Buckets are a documented judgment call.
Regenerate after refreshing corpus_dataset.csv (e.g. when giants land)."""
import csv, sys
SC=sys.argv[1] if len(sys.argv)>1 else '/home/edward/Projects/StoryDaemon/work/corpus/scores'
rows={r['book']:r for r in csv.DictReader(open(f'{SC}/corpus_dataset.csv'))}
BUCKETS=[
 ('Adventure / thriller',['sabatini-captainblood','buchan-thirtyninesteps','buchan-greenmantle','sabatini-scaramouche','haggard-ksm','haggard-she','childers-riddlesands']),
 ('Sci-fi & fantasy',['eddison-ouroboros','wells-warofworlds','wells-timemachine','dunsany-elfland']),
 ('Horror / gothic',['stoker-dracula','bronte-janeeyre']),
 ('Mystery / ironic / psychological',['collins-moonstone','conrad-secretagent','conrad-heartofdarkness','conrad-lordjim','dickens-taleoftwocities','collins-womaninwhite']),
 ('Domestic / social realism',['austen-emma','austen-persuasion','austen-pride','eliot-middlemarch']),
]  # giants (womaninwhite, middlemarch) added to buckets here; they populate once in corpus_dataset.csv
def fl(b,k):
    try: return float(rows[b][k])
    except: return None
def m(bl,k): v=[fl(b,k) for b in bl if fl(b,k) is not None]; return sum(v)/len(v) if v else float('nan')
print('| Genre bucket | n | Tension | Peak pos | Dialogue | Calm% | High% | Threads |')
print('|---|---|---|---|---|---|---|---|')
for g,bl in BUCKETS:
    bl=[b for b in bl if fl(b,'nd1_tension_mean') is not None]
    if not bl: continue
    print('| %s | %d | %.2f | %.2f | %.0f%% | %.0f%% | %.0f%% | %.1f |'%(g,len(bl),m(bl,'nd1_tension_mean'),m(bl,'nd1_tension_peak_position'),100*m(bl,'nd1_br_dialogue_share'),100*m(bl,'nd1_tension_calm_share'),100*m(bl,'nd1_tension_high_share'),m(bl,'nd1_th_threads_total')))
