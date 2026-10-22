import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from email.message import Message

from protego import Protego
from crawl_http import Client, CrawlError, StopCrawl, normalize, retry_delay
from crawler import crawl, load_config, parse_sitemap
from extract_content import extract_page

BASE = 'https://www.fnx.co.il'


def config():
    return load_config(Path(__file__).resolve().parents[1] / 'crawl_config.json')


class Response(io.BytesIO):
    status = 200
    def __init__(self, body, content_type='text/html; charset=utf-8'):
        super().__init__(body)
        self.headers = {'Content-Type': content_type}


class Opener:
    def __init__(self, routes):
        self.routes, self.calls = routes, []
    def open(self, req, timeout):
        self.calls.append(req.full_url)
        item = self.routes[req.full_url]
        if isinstance(item, list):
            item = item.pop(0)
        if isinstance(item, Exception):
            raise item
        return Response(item)


def error(url, code, **headers):
    h = Message()
    for k, v in headers.items():
        h[k] = v
    return urllib.error.HTTPError(url, code, 'test', h, None)


def client(routes):
    opener = Opener(routes)
    c = Client(config(), opener=opener, clock=lambda: 0, sleeper=lambda _: None)
    c.policy = Protego.parse('User-agent: *\nDisallow: /blocked\nDisallow: /*?page=')
    return c, opener


class TransportTests(unittest.TestCase):
    def test_normalization_and_query_not_silently_changed(self):
        self.assertEqual(normalize(BASE + '/a#x'), BASE + '/a')
        self.assertEqual(normalize(BASE + '/%61/'), BASE + '/a/')
        for u in [BASE + '/a?page=2', BASE + ':444/a', BASE + '.evil/a', 'https://x@www.fnx.co.il/a', 'https://www.fnx.co.il/a%5Cb']:
            self.assertIsNone(normalize(u))

    def test_modern_robots_wildcard_and_allow_precedence(self):
        p = Protego.parse('User-agent: *\nDisallow: /*?page=\nDisallow: /private\nAllow: /private/public$')
        self.assertFalse(p.can_fetch(BASE + '/news?page=2', 'PhoenixRAGPrototype'))
        self.assertFalse(p.can_fetch(BASE + '/private/a', 'PhoenixRAGPrototype'))
        self.assertTrue(p.can_fetch(BASE + '/private/public', 'PhoenixRAGPrototype'))

    def test_blocked_redirect_never_fetched(self):
        for target in [BASE + '/blocked/a', 'https://example.com/', BASE + '/login']:
            c, o = client({BASE + '/start': error(BASE + '/start', 302, Location=target)})
            with self.assertRaises(CrawlError): c.fetch(BASE + '/start')
            self.assertEqual(o.calls, [BASE + '/start'])

    def test_redirect_loop_bounded(self):
        c, o = client({BASE + '/a': error(BASE + '/a', 302, Location='/b'), BASE + '/b': error(BASE + '/b', 302, Location='/a')})
        with self.assertRaisesRegex(CrawlError, 'redirect_loop'): c.fetch(BASE + '/a')
        self.assertEqual(len(o.calls), 2)

    def test_retry_recovers_and_honors_retry_after(self):
        c, o = client({BASE + '/a': [error(BASE + '/a', 429, **{'Retry-After': '4'}), b'ok']})
        waits = []; c.sleep = waits.append
        self.assertEqual(c.fetch(BASE + '/a')[0], b'ok')
        self.assertEqual(len(o.calls), 2)
        self.assertGreaterEqual(waits[-1], 4)

    def test_long_cooldown_stops_entire_run(self):
        c, o = client({BASE + '/a': error(BASE + '/a', 429, **{'Retry-After': '120'})})
        with self.assertRaises(StopCrawl): c.fetch(BASE + '/a')
        self.assertEqual(len(o.calls), 1)

    def test_request_and_body_limits(self):
        c, o = client({BASE + '/a': b'123456789'})
        c.c['max_response_bytes'] = 4
        with self.assertRaisesRegex(CrawlError, 'response_too_large'): c.fetch(BASE + '/a')
        c.c['max_requests'] = 1
        with self.assertRaises(StopCrawl): c.fetch(BASE + '/a')
        self.assertEqual(len(o.calls), 1)

    def test_retry_after_http_date(self):
        from datetime import datetime, timezone
        self.assertEqual(retry_delay('Thu, 15 Oct 2026 08:00:10 GMT', datetime(2026,10,15,8,0,0,tzinfo=timezone.utc)), 10)

    def test_missing_robots_stops_before_any_pages(self):
        c, o = client({BASE + '/robots.txt': error(BASE + '/robots.txt', 403)})
        with tempfile.TemporaryDirectory() as folder:
            path, report = crawl(c.c, folder, c)
            self.assertEqual(report['status'], 'failed')
            self.assertTrue((path / 'manifest.json').exists())
        self.assertEqual(o.calls, [BASE + '/robots.txt'])


class ExtractionTests(unittest.TestCase):
    def test_main_preserves_disclaimer_faq_table_and_link(self):
        html = '''<html><body><nav>תפריט אתר גלובלי</nav><main><h1>כותרת</h1>
        <p>הסבר ראשון שני שלישי רביעי חמישי שישי שביעי שמיני תשיעי עשירי נוסף מידע חשוב ברור</p>
        <p>הכיסוי כפוף לתנאי הפוליסה.</p><details><summary>מה צריך להביא?</summary><p>אישור משטרה</p></details>
        <table><tr><th>מסמך</th><th>מחיר</th></tr><tr><td>אישור</td><td>150</td></tr></table>
        <a href="/form.pdf">טופס תביעה</a><script>סוד</script></main><footer>פוטר גלובלי</footer></body></html>'''
        r = extract_page(html, BASE + '/a')
        self.assertEqual(r['status'], 'extracted')
        for phrase in ['הכיסוי כפוף לתנאי הפוליסה.', 'מה צריך להביא?', 'אישור משטרה', '| אישור | 150 |', BASE + '/form.pdf']:
            self.assertIn(phrase, r['text'])
        for phrase in ['תפריט אתר גלובלי', 'פוטר גלובלי', 'סוד']:
            self.assertNotIn(phrase, r['text'])

    def test_absent_or_ambiguous_main_requires_review(self):
        for html in ['<body>information</body>', '<main>one</main><main>two</main>']:
            self.assertEqual(extract_page(html, BASE)['status'], 'needs_review')

    def test_sitemap_ignores_image_extension_and_rejects_entities(self):
        raw = b'<urlset xmlns:image="http://example.com"><url><loc>https://www.fnx.co.il/a</loc><image:image><image:loc>https://x/p.png</image:loc></image:image></url></urlset>'
        self.assertEqual(parse_sitemap(raw, 1000)[1], [BASE + '/a'])
        with self.assertRaises(CrawlError): parse_sitemap(b'<!DOCTYPE x [<!ENTITY a "x">]><urlset/>', 1000)

    def test_nested_maps_duplicates_and_page_limit(self):
        c = config(); c.update(max_pages=1, seeds=[])
        html = ('<main><h1>מידע</h1><p>אחד שני שלישי רביעי חמישי שישי שביעי שמיני תשיעי עשירי נוסף אחר מסמך תביעה כיסוי</p><a href="/new">המשך</a><a href="/form.pdf">טופס</a></main>').encode()
        routes = {BASE+'/robots.txt': b'User-agent: *\nSitemap: https://www.fnx.co.il/index.xml',
                  BASE+'/index.xml': b'<sitemapindex><sitemap><loc>https://www.fnx.co.il/map.xml</loc></sitemap><sitemap><loc>https://www.fnx.co.il/index.xml</loc></sitemap></sitemapindex>',
                  BASE+'/map.xml': b'<urlset><url><loc>https://www.fnx.co.il/a</loc></url><url><loc>https://www.fnx.co.il/a#part</loc></url><url><loc>https://www.fnx.co.il/b</loc></url></urlset>', BASE+'/a': html}
        op = Opener(routes); cl=Client(c,opener=op,clock=lambda:0,sleeper=lambda _:None)
        with tempfile.TemporaryDirectory() as folder:
            path, report = crawl(c,folder,cl)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['page_attempts'],1)
            self.assertEqual(report['remaining_queue'],2)
            self.assertEqual(report['discovery_status_counts']['pdf_deferred'],1)
            self.assertEqual(len(report['sitemaps']),2)
            self.assertEqual(len((path/'pages.jsonl').read_text().splitlines()),1)
        self.assertEqual(op.calls.count(BASE+'/index.xml'),1)

if __name__ == '__main__':
    unittest.main()
