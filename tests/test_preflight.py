import unittest
from preflight import in_scope, sitemap_urls, VisibleText, ScopedRedirect

class PreflightTests(unittest.TestCase):
    def test_scope_rejects_external_and_credentials(self):
        for url in ['https://www.fnx.co.il.evil.test/', 'http://www.fnx.co.il/', 'https://user@www.fnx.co.il/', 'https://www.fnx.co.il:444/']:
            self.assertFalse(in_scope(url))
        self.assertTrue(in_scope('https://www.fnx.co.il/apartment-insurance/'))

    def test_diagnostic_does_not_mistake_script_for_page_text(self):
        p = VisibleText()
        p.feed('<script>מידע נסתר</script><style>hidden</style><main><h1>ביטוח</h1><p>טקסט גלוי</p></main>')
        self.assertEqual(p.parts, ['ביטוח', 'טקסט גלוי'])

    def test_sitemap_namespace_and_index(self):
        kind, urls = sitemap_urls('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://www.fnx.co.il/a.xml</loc></sitemap></sitemapindex>')
        self.assertEqual(kind, 'sitemapindex')
        self.assertEqual(urls, ['https://www.fnx.co.il/a.xml'])

    def test_external_redirect_rejected(self):
        with self.assertRaises(ValueError):
            ScopedRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com/')

if __name__ == '__main__':
    unittest.main()
