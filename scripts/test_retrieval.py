from sentence_transformers import SentenceTransformer

import retrieval_pipeline as R

encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

query = 'What is the treatment for atrial fibrillation?'

result = R.hierarchical_retrieve(query, encoder)

print('Query:', query)

print()

print('=== L2 Textbook chunks ===')

for _, row in result['l2_chunks'].iterrows():

    print(f'  [{row.score:.3f}] {row.title}: {row.content[:100]}')

print()

print('=== L3 PubMed chunks ===')

for _, row in result['l3_chunks'].iterrows():

    print(f'  [{row.score:.3f}] {row.title}: {row.content[:100]}')