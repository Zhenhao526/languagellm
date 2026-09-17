"""Network-free checks for acquisition stop rules and byte accounting."""
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import download_originals as d


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, url, declared=None):
        super().__init__(body)
        self.url = url
        self.headers = {} if declared is None else {'Content-Length': str(declared)}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        (self.out / 'plan.json').write_text('{}')
        self.jobs = [dict(request_index=i, pageid=i, review_id=f'V_{i}',
                          original_url=f'https://upload.wikimedia.org/test/{i}',
                          size=4, original_file_sha1=hashlib.sha1(b'abcd').hexdigest()) for i in range(2)]
        self.plan = {'stage': 'water', 'jobs': self.jobs}

    def run_fake(self, opener):
        with patch.object(d, 'verify', return_value=self.plan), patch.object(d.time, 'sleep'):
            return d.execute(self.out, opener)

    def test_429_stops_whole_round_and_archives_error(self):
        error = HTTPError(self.jobs[0]['original_url'], 429, 'Too Many Requests',
                          {'Retry-After': '60'}, io.BytesIO(b'limited'))
        opener = Opener([error])
        result = self.run_fake(opener)
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(result['requests'], 1)
        self.assertEqual(result['not_requested_pageids'], [1])
        self.assertEqual(result['response_bytes'], 7)
        self.assertEqual(result['status'], 'stopped_on_error')
        receipt = d.read(self.out / 'execution/receipts/000_receipt.json')
        self.assertEqual(receipt['response_headers']['Retry-After'], '60')

    def test_success_and_sha_mismatch_have_distinct_status(self):
        opener = Opener([Response(b'abcd', self.jobs[0]['original_url'], 4),
                         Response(b'abce', self.jobs[1]['original_url'], 4)])
        result = self.run_fake(opener)
        self.assertEqual(result['downloaded'], 1)
        self.assertEqual(result['metadata_mismatch_rejected'], 1)
        self.assertEqual(result['response_bytes'], 8)
        self.assertEqual(result['status'], 'download_stage_complete')
        self.assertEqual(len(opener.calls), 2)

    def test_oversize_header_stops_without_reading_body(self):
        response = Response(b'abcd', self.jobs[0]['original_url'], d.MAX_FILE + 1)
        result = self.run_fake(Opener([response]))
        self.assertEqual(result['response_bytes'], 0)
        self.assertEqual(result['requests'], 1)
        self.assertEqual(result['status'], 'stopped_on_error')

    def test_early_network_error_never_retries(self):
        opener = Opener([TimeoutError('test')])
        result = self.run_fake(opener)
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(result['response_bytes'], 0)
        self.assertEqual(result['not_requested_pageids'], [1])

    def test_stream_byte_cap_and_exact_original_preservation(self):
        body = io.BytesIO(b'abcdefgh')
        path = self.out / 'bounded.bin'
        count, eof = d.stream_body(body, path, 4, monotonic=lambda: 0)
        self.assertEqual((count, eof), (4, False))
        self.assertEqual(body.tell(), 4)
        self.assertEqual(path.read_bytes(), b'abcd')

    def test_declared_length_is_also_a_read_ceiling(self):
        self.plan['jobs'] = self.jobs[:1]
        response = Response(b'abcdEXTRA', self.jobs[0]['original_url'], 4)
        result = self.run_fake(Opener([response]))
        self.assertEqual(result['response_bytes'], 4)
        self.assertEqual(result['downloaded'], 1)
        self.assertEqual((self.out / 'execution/originals/V_0.bin').read_bytes(), b'abcd')

    def test_shorter_than_declared_response_stops(self):
        response = Response(b'ab', self.jobs[0]['original_url'], 4)
        result = self.run_fake(Opener([response]))
        self.assertEqual(result['requests'], 1)
        self.assertEqual(result['response_bytes'], 2)
        self.assertEqual(result['status'], 'stopped_on_error')

    def test_no_redirect_and_no_execution_overwrite(self):
        self.assertIsNone(d.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com'))
        (self.out / 'execution').mkdir()
        with patch.object(d, 'verify', return_value=self.plan):
            with self.assertRaises(FileExistsError):
                d.execute(self.out, Opener([]))


if __name__ == '__main__':
    unittest.main()
