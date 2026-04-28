# scripts/test_retrieval.py
from sentence_transformers import SentenceTransformer
import retrieval_pipeline as R

encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Test queries with expected domain
test_queries = [
    ("What is the treatment for atrial fibrillation?",        "Disorders"),
    ("How is type 2 diabetes managed?",                        "Disorders"),
    ("What is the mechanism of action of warfarin?",           "Chemicals & Drugs"),
    ("Describe the anatomy of the brachial plexus.",           "Anatomy"),
    ("What are the steps in a laparoscopic cholecystectomy?",  "Procedures"),
    ("How does the renin-angiotensin system regulate BP?",     "Physiology"),
    ("What gene mutations cause familial hypercholesterolaemia?", "Physiology"),
]

print("=" * 80)
print("DOMAIN CLASSIFIER + RETRIEVAL PIPELINE TEST")
print("=" * 80)

correct_domain = 0
for query, expected_domain in test_queries:
    result = R.hierarchical_retrieve(query, encoder)

    domain_ok = result["domain"] == expected_domain
    correct_domain += int(domain_ok)

    print(f"\nQuery: {query}")
    print(f"  Domain     : {result['domain']} (expected: {expected_domain}) {'✓' if domain_ok else '✗'}")
    print(f"  Confidence : {result['confidence']:.3f}")
    print(f"  Routed to  : {result['source_route']}")
    print(f"  L2 Textbook chunks:")
    for _, row in result["l2_chunks"].iterrows():
        print(f"    [{row.score:.3f}] {row.title}: {row.content[:100]}")
    print(f"  L3 PubMed chunks:")
    for _, row in result["l3_chunks"].iterrows():
        print(f"    [{row.score:.3f}] {row.title}: {row.content[:100]}")

print("\n" + "=" * 80)
print(f"Domain classification accuracy: {correct_domain}/{len(test_queries)} = {correct_domain/len(test_queries):.1%}")
print("=" * 80)