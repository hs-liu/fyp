from src.scripts.rag.retrieval_pipeline import name_to_cui, G, CLINICAL_PRIORITY_STYS

# Check what clinical names match this question
q = 'a 65-year old man presents with gradually worsening rigidity of his arms and legs and slowness in performing tasks. he says he has also noticed hand tremors which increase at rest and decrease with focused movements. on examination the patient does not swing his arms while walking and has a shortened shuffling gait. an antiviral drug is prescribed which alleviates the patients symptoms'

matches = []
for name, cui in name_to_cui.items():
    if name in q:
        sty = G.nodes[cui].get('sty_name', '') if cui in G else ''
        if sty in CLINICAL_PRIORITY_STYS:
            matches.append((name, cui, sty))

matches.sort(key=lambda x: len(x[0]), reverse=True)
print(f'Clinical matches: {len(matches)}')
for name, cui, sty in matches[:20]:
    print(f'  {name!r} -> {cui} [{sty}]')
