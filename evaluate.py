"""Batch evaluation CLI for the Customer Support RAG system.

Example usage:
  python evaluate.py --queries "Password reset not working" "I want a refund" --top-k 3
  python evaluate.py --query-file queries.txt --json
  python evaluate.py --dry-run

If real API keys are absent or --dry-run is provided, the script produces structural output without calling external services.
"""
from __future__ import annotations
import argparse, json, time
from typing import List, Dict, Any
import config  # ensures env vars loaded
from modules.vector_store import VectorStore
from modules.response_generator import ResponseGenerator
from modules.sentiment_analysis import SentimentAnalyzer
from modules.evaluation import RAGEvaluator
from modules.knowledge_processor import KnowledgeProcessor

def have_real_keys() -> bool:
    return (
        config.OPENAI_API_KEY and config.OPENAI_API_KEY != "default_openai_key" and
        config.PINECONE_API_KEY and config.PINECONE_API_KEY != "default_pinecone_key"
    )

def load_queries(args) -> List[str]:
    if args.query_file:
        with open(args.query_file, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    if args.queries:
        return args.queries
    # Default canned queries
    return [
        "I can't reset my password and I'm locked out",
        "I was charged twice this month, need a refund",
        "How do I integrate the API and what are the rate limits?",
        "My account was suspended without warning"
    ]

def ensure_kb(vector_store: VectorStore):
    try:
        stats = vector_store.get_index_stats()
        if not stats or stats.get('total_vectors', 0) == 0:
            KnowledgeProcessor().load_knowledge_base()
    except Exception:
        pass

def evaluate(queries: List[str], top_k: int, dry_run: bool) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    t_all = time.time()
    if dry_run or not have_real_keys():
        for q in queries:
            results.append({
                'query': q,
                'dry_run': True,
                'metrics': {k: 0.0 for k in ['context_precision','context_recall','faithfulness','answer_relevancy','retrieval_accuracy']}
            })
        return {'dry_run': True, 'results': results, 'overall_average': 0.0}
    vector_store = VectorStore()
    ensure_kb(vector_store)
    responder = ResponseGenerator()
    sentiment = SentimentAnalyzer()
    evaluator = RAGEvaluator()
    history: List[Dict[str, Any]] = []
    for q in queries:
        t0 = time.time()
        retrieved = vector_store.similarity_search(q, k=top_k)
        sent = sentiment.analyze_sentiment(q)
        resp = responder.generate_response(q, retrieved, history, sent)
        eval_res = evaluator.evaluate_response(q, resp['response'], retrieved)
        history.append({'sender': 'customer', 'content': q, **sent})
        history.append({'sender': 'agent', 'content': resp['response']})
        results.append({
            'query': q,
            'response': resp['response'],
            'retrieved_docs': [d.get('title','') for d in retrieved],
            'overall_score': eval_res['overall_score'],
            'metrics': eval_res['metrics'],
            'latency_s': round(time.time() - t0, 2)
        })
    overall_avg = sum(r['overall_score'] for r in results) / len(results)
    return {
        'dry_run': False,
        'query_count': len(results),
        'overall_average': overall_avg,
        'results': results,
        'total_time_s': round(time.time() - t_all, 2)
    }

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate RAG system responses.')
    parser.add_argument('--queries', nargs='*', help='Inline queries')
    parser.add_argument('--query-file', help='Path to file with one query per line')
    parser.add_argument('--top-k', type=int, default=3, help='Top K documents to retrieve')
    parser.add_argument('--json', action='store_true', help='Output raw JSON only')
    parser.add_argument('--dry-run', action='store_true', help='Skip external API calls')
    args = parser.parse_args()
    dry_run = args.dry_run
    if not dry_run and not have_real_keys():
        print('⚠️  Real API keys not found; running in dry-run mode.')
        dry_run = True
    queries = load_queries(args)
    report = evaluate(queries, args.top_k, dry_run)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print('\n=== Evaluation Report ===')
    print(f"Mode: {'DRY-RUN' if report.get('dry_run') else 'LIVE'}")
    print(f"Queries: {len(report.get('results', []))}")
    if not report.get('dry_run'):
        print(f"Overall Average Score: {report.get('overall_average', 0):.3f}")
    for r in report['results']:
        print('\n---')
        print('Query:', r['query'])
        if not report.get('dry_run'):
            print('Retrieved:', ', '.join(r['retrieved_docs']))
            print('Score:', f"{r['overall_score']:.3f}", '| Latency:', f"{r['latency_s']}s")
            m = r['metrics']
            print('Metrics:', f"prec={m['context_precision']:.2f} faith={m['faithfulness']:.2f} rel={m['answer_relevancy']:.2f} retr={m['retrieval_accuracy']:.2f}")
    print('\nDone.')

if __name__ == '__main__':
    main()
