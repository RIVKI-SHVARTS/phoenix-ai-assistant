"""Source-preserving extraction; heuristic extraction is an audit, not authority."""
import hashlib
import re
from collections import Counter
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag
import trafilatura

VERSION = 'main-dom-v1'
DROP = 'script,style,noscript,svg,iframe,input,select,textarea,nav.breadcrumbs,nav.sticky-nav'
BLOCKS = {'div', 'section', 'article', 'main', 'header', 'footer', 'p', 'ul', 'ol', 'dl', 'dt', 'dd', 'figure', 'figcaption', 'form', 'details', 'summary'}


def clean_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def tokens(text):
    return Counter(re.findall(r'[\u0590-\u05ff]+|\d+', text))


def token_coverage(source, extracted):
    a, b = tokens(source), tokens(extracted)
    return sum((a & b).values()) / max(1, sum(a.values()))


def render(node, url):
    if isinstance(node, NavigableString):
        return re.sub(r'\s+', ' ', str(node))
    if not isinstance(node, Tag):
        return ''
    name = node.name
    if name == 'br':
        return '\n'
    if name == 'table':
        rows = []
        for row in node.find_all('tr'):
            cells = [clean_spaces(''.join(render(child, url) for child in c.children)).replace('|', '\\|') for c in row.find_all(['th', 'td'], recursive=False)]
            if cells:
                rows.append('| ' + ' | '.join(cells) + ' |')
        caption = node.find('caption')
        return '\n\n' + (caption.get_text(' ', strip=True) + '\n' if caption else '') + '\n'.join(rows) + '\n\n'
    body = ''.join(render(child, url) for child in node.children)
    if name == 'a':
        href = urljoin(url, node.get('href', ''))
        label = clean_spaces(body)
        if label and node.get('href') and urlsplit(href).scheme in ('https', 'http', 'mailto', 'tel'):
            return '[' + label + '](' + href.replace(' ', '%20').replace('(', '%28').replace(')', '%29') + ') '
        return body
    if re.fullmatch('h[1-6]', name):
        return '\n\n' + '#' * int(name[1]) + ' ' + clean_spaces(body) + '\n\n'
    if name == 'li':
        return '\n- ' + body.strip() + '\n'
    if name in BLOCKS:
        return '\n\n' + body.strip() + '\n\n'
    return ' ' + body + ' '


def extract_page(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    title = clean_spaces(soup.title.get_text(' ', strip=True)) if soup.title else url
    # Keep all discovered links for crawler classification; never submit forms.
    discovered = list(dict.fromkeys(urljoin(url, a['href']) for a in soup.select('a[href]')))
    mains = soup.select('main') or soup.select('[role=main]')
    warnings = []
    if len(mains) != 1:
        # Do not silently treat the complete body/navigation as article content.
        return {'url': url, 'title': title, 'status': 'needs_review', 'warnings': ['missing_or_ambiguous_main'], 'discovered_links': discovered}
    main = mains[0]
    for node in list(main.select(DROP)):
        if node.parent:
            node.decompose()
    # HTML comments are not user-facing information.
    from bs4 import Comment
    for comment in main.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    source = main.get_text(' ', strip=True)
    markdown = render(main, url)
    markdown = re.sub(r'[ \t]+\n', '\n', markdown)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown).strip()
    heuristic = trafilatura.extract('<html><body>' + str(main) + '</body></html>', url=url, output_format='markdown', include_comments=False,
                                   include_tables=True, include_links=True, favor_recall=True) or ''
    coverage = token_coverage(source, markdown)
    heuristic_coverage = token_coverage(source, heuristic)
    if coverage < .995:
        warnings.append('source_token_loss')
    if len(tokens(source)) < 15:
        warnings.append('very_short_content')
    if heuristic_coverage < .95:
        warnings.append('heuristic_extractor_would_omit_content')
    # Retain headings/paragraph boundaries and link labels; attach metadata later.
    sections = []
    heading, lines = title, []
    for line in markdown.splitlines():
        if re.match(r'^#{1,6} ', line):
            if '\n'.join(lines).strip() or heading != title:
                sections.append({'heading': heading, 'text': '\n'.join(lines).strip()})
            heading, lines = line.lstrip('# ').strip(), []
        else:
            lines.append(line)
    if '\n'.join(lines).strip() or heading != title:
        sections.append({'heading': heading, 'text': '\n'.join(lines).strip()})
    headings = [clean_spaces(x.get_text(' ', strip=True)) for x in main.select('h1,h2,h3,h4,h5,h6')]
    return {'url': url, 'title': title, 'status': 'extracted' if coverage >= .995 and len(tokens(source)) >= 15 else 'needs_review',
            'text': markdown, 'sections': sections, 'headings': headings,
            'content_hash': hashlib.sha256(markdown.encode()).hexdigest(),
            'extractor_version': VERSION, 'warnings': warnings,
            'quality': {'main_text_chars': len(source), 'extracted_chars': len(markdown),
                        'source_token_coverage': round(coverage, 6), 'trafilatura_token_coverage': round(heuristic_coverage, 6)},
            'discovered_links': discovered}
