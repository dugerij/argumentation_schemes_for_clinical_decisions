import argparse
import json

from dotenv import load_dotenv

from entity_extraction.umls import UMLSClient, UMLSConfig
from helpers.jsonl import to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Look up clinical terms in UMLS.")
    parser.add_argument("terms", nargs="+", help="Clinical terms to search.")
    parser.add_argument("--sources", default=None, help="Comma-separated UMLS source vocabularies.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    sources = tuple(item.strip() for item in args.sources.split(",")) if args.sources else None
    client = UMLSClient(UMLSConfig.from_env())

    results = {
        term: client.search(term, source_vocabularies=sources)
        for term in args.terms
    }
    print(json.dumps(to_jsonable(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
