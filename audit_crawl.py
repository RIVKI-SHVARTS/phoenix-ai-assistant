"""Offline quality checks against the saved original HTML (no network or AI)."""
import argparse
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
from extract_content import DROP


def normalized(text):
    text = re.sub(r'\[([^\]]*)\]\([^\n]*?\)', r'\1', text)
    return re.sub(r'[^\w\u0590-\u05ff]', '', text)


CRITICAL = {
 '/apartment-insurance/': ['ביטוח לעובדים אקראיים או קבועים', 'תנאי הפוליסה גוברים'],
 '/claims-customer-service/claims-appartment-insurance/damage-to-the-apartment/': ['תמונות ממקום האירוע', 'פרטי חשבון בנק', 'אישור משטרה', 'יודגש, כי אין במפורט בהנחיות אלו כדי להוות הכרה בכיסוי הביטוחי', 'rechush@fnx.co.il'],
 '/claims-customer-service/claims-car-insurance/claims-property-car-insurance/': ['לצרף את המסמכים', 'יודגש, כי אין במפורט בהנחיות אלו כדי להוות הכרה בכיסוי הביטוחי'],
 '/claims-customer-service/claims-appartment-insurance/claims-third-party-house/': ['הגשת תביעה כללית', 'יודגש, כי אין במפורט בהנחיות אלו כדי להוות הכרה בכיסוי הביטוחי'],
 '/customer-service/': ['הגשת תביעת רכב', 'הגשת תביעת דירה', 'איתור שרברב'],
 '/apartment-insurance/apartment-contents-insurance/': ['ביטוח חבות מעבידים לעובדי משק הבית', 'תנאי הפוליסה גוברים', 'פריטים יקרי ערך']
}


def audit(run):
    results = []
    for row in (run / 'pages.jsonl').read_text(encoding='utf-8').splitlines():
        page = json.loads(row)
        soup = BeautifulSoup((run / page['raw_file']).read_bytes(), 'html.parser')
        main = soup.find('main')
        if main is None:
            results.append({'url': page['url'], 'pass': False, 'reason': 'missing_main'})
            continue
        for node in list(main.select(DROP)):
            if node.parent: node.decompose()
        original = main.get_text(' ', strip=True)
        text = page.get('text', '')
        rendered = normalized(text)
        # Independent sequence check: not merely a bag-of-words score.
        blocks = []
        for node in main.select('h1,h2,h3,h4,h5,h6,p,li,td,th,summary'):
            fragment = node.get_text(' ', strip=True)
            if len(fragment) >= 20:
                blocks.append(fragment)
        blocks = list(dict.fromkeys(blocks))
        missing = [b for b in blocks if normalized(b) not in rendered]
        from urllib.parse import urlsplit
        phrases = CRITICAL.get(urlsplit(page['url']).path, [])
        phrase_checks = [{'phrase': p, 'present_in_original': normalized(p) in normalized(original), 'present_in_extraction': normalized(p) in rendered} for p in phrases]
        global_noise = [p for p in ['דלג לתפריט עליון', 'פורטל מעסיקים', 'Investor Relations'] if p in text]
        results.append({'url': page['url'], 'title': page['title'], 'pass': not missing and not global_noise and all(p['present_in_original'] and p['present_in_extraction'] for p in phrase_checks),
                        'blocks_checked': len(blocks), 'missing_blocks': missing,
                        'critical_checks': phrase_checks, 'global_navigation_leaks': global_noise,
                        'table_count_in_main': len(main.select('table')), 'quality': page.get('quality', {})})
    report = {'all_passed': bool(results) and all(x['pass'] for x in results), 'pages': results,
              'limitations': ['Tests use saved HTML, not browser-rendered content.', 'Token/block retention is not proof of semantic correctness.', 'No real HTML table occurred in these six pages; table preservation is covered by a synthetic unit test.', 'No AI answers or full-site coverage evaluated.']}
    (run / 'quality-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    r=audit(args.run)
    print(json.dumps({'all_passed':r['all_passed'],'pages':len(r['pages']),'blocks':sum(x.get('blocks_checked',0) for x in r['pages']), 'missing':sum(len(x.get('missing_blocks',[])) for x in r['pages'])},ensure_ascii=False))
    raise SystemExit(0 if r['all_passed'] else 1)
