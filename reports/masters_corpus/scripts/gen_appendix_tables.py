import csv, os
SC='/home/edward/Projects/StoryDaemon/work/corpus/scores'
rows=list(csv.DictReader(open(f'{SC}/corpus_dataset.csv')))
def author(b): return b.split('-')[0]
rows.sort(key=lambda r:(author(r['book']), r['book']))
frac=('_share','_ratio_mean','_rate','_position','_sim','distinct_3_ratio','self_bleu_mean',
      'intra_unigram','intra_bigram','intra_trigram','_self_transition','_volatility','_std')
def cell(r,k):
    v=r.get(k,'')
    if v is None or v=='': return '—'
    try:
        f=float(v)
        if any(k.endswith(s) for s in frac): return f'{f:.3f}'
        if f==int(f): return str(int(f))
        return f'{f:g}'
    except: return v
def table(title, cols, subset=None):
    hdr=['author','book']+[c[1] for c in cols]
    out=[f'\n**{title}**\n', '| '+' | '.join(hdr)+' |', '|'+'|'.join(['---']*len(hdr))+'|']
    prev=None
    for r in rows:
        if subset and not r.get(subset,'').strip(): continue
        a=author(r['book']); ad=a if a!=prev else ''; prev=a
        out.append('| '+' | '.join([ad, r['book'].split('-',1)[1]]+[cell(r,c[0]) for c in cols])+' |')
    return '\n'.join(out)
P=[]
P.append("## Appendix A: Full metric tables\n")
P.append("*Every metric computed, straight from `corpus_dataset.csv`. Grouped by author; "
"one table per metric family. Both st1 and nd1 now cover all 26 books. See Sections 2 and 5-9 for "
"what each metric means and how to read it; the caveats in Section 12 apply "
"(cast counts are not coreference-merged; opening-formula is title-contaminated).*")
P.append("\n### A.1 st1 (deterministic, 26 books)")
P.append(table('Size & lexical', [('st1_total_words','words'),('st1_mean_chapter_words','words/chapter'),('st1_mtld_mean','MTLD'),('st1_mtld_unreliable_runs','MTLD unreliable'),('st1_burstiness_mean','burstiness')]))
P.append(table('Craft & style', [('st1_cliche_per_1k','cliche/1k'),('st1_slop_per_1k','slop/1k'),('st1_em_dash_per_1k','em-dash/1k'),('st1_dialogue_ratio_mean','dialogue ratio')]))
P.append(table('Diversity & self-repetition', [('st1_distinct_3_ratio','distinct-3'),('st1_self_bleu_mean','self-BLEU'),('st1_intra_unigram','intra-uni'),('st1_intra_bigram','intra-bi'),('st1_intra_trigram','intra-tri')]))
P.append(table('Duplication', [('st1_duplication_suspected','dup suspected'),('st1_max_pairwise_sim','max pair sim'),('st1_max_verbatim_chars','max verbatim'),('st1_n_flagged_pairs','flagged pairs')]))
P.append(table('Openings & cast', [('st1_opening_mean_pairwise','opening sim'),('st1_opening_high_pair_rate','opening hi-rate'),('st1_cast_size','cast size'),('st1_recurring_cast_size','recurring cast'),('st1_person_mentions_per_1k','mentions/1k')]))
P.append("\n### A.2 nd1 (LLM-judged, all 26 books)")
P.append(table('Tension: level', [('nd1_tension_mean','mean'),('nd1_tension_std','std'),('nd1_tension_min','min'),('nd1_tension_max','max'),('nd1_tension_peak','peak'),('nd1_tension_peak_position','peak pos')], subset='nd1_tension_mean'))
P.append(table('Tension: shape', [('nd1_tension_calm_share','calm share'),('nd1_tension_high_share','high share'),('nd1_tension_volatility','volatility'),('nd1_tension_tail_mean','tail mean'),('nd1_tension_tail_final','final')], subset='nd1_tension_mean'))
P.append(table('Block rhythm: mode shares', [('nd1_br_setting_share','setting'),('nd1_br_character_desc_share','char-desc'),('nd1_br_lore_share','lore'),('nd1_br_dialogue_share','dialogue'),('nd1_br_action_share','action'),('nd1_br_interiority_share','interior'),('nd1_br_transition_share','transit')], subset='nd1_tension_mean'))
P.append(table('Block rhythm: structural gauges', [('nd1_br_switch_rate','switch rate'),('nd1_br_words_per_mode_segment','words/segment'),('nd1_br_interiority_self_transition','interior self-trans'),('nd1_br_secondary_shading_rate','2ndary shading'),('nd1_br_setting_touch_rate','setting touch')], subset='nd1_tension_mean'))
P.append(table('Thread architecture', [('nd1_th_threads_total','threads'),('nd1_th_threads_2plus','threads 2+'),('nd1_th_switch_rate','switch rate'),('nd1_th_run_length_mean','run mean'),('nd1_th_run_length_max','run max'),('nd1_th_convergence_events','converge'),('nd1_th_first_convergence_position','1st converge pos')], subset='nd1_tension_mean'))
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'appendix_tables.md')
open(OUT,'w').write('\n'.join(P)+'\n')
print('wrote', OUT)
print('lines:', len(open(OUT).read().splitlines()))
print('tables:', '\n'.join(l for l in open(OUT).read().splitlines() if l.startswith('**')))
