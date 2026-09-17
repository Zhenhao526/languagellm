"""Offline regression checks; fake HTTP errors, temporary logs, no network."""
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from email.message import Message
from email.utils import format_datetime
import datetime as dt
import hashlib
import importlib.util
import io
import json
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('metadata_collector_check', HERE / 'collect_metadata.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = []
    cases = [('lowercase_seconds', '34', 429),
             ('http_date', format_datetime(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=45), usegmt=True), 429),
             ('long_deferral', '120', 429), ('access_refusal', None, 403)]
    for name, retry, code in cases:
        headers = Message()
        if retry is not None:
            headers['retry-after'] = retry
        errors = [HTTPError('https://commons.wikimedia.org/w/api.php', code, 'offline test', headers,
                            io.BytesIO(b'offline simulated response')) for _ in range(2)]
        waits = []
        with tempfile.TemporaryDirectory() as temporary:
            mod.HERE = Path(temporary); (mod.HERE / 'raw').mkdir()
            with patch.object(mod, 'urlopen', side_effect=errors) as request, \
                 patch.object(mod.time, 'sleep', side_effect=lambda seconds: waits.append(seconds)):
                client = mod.Client(interval=10, resume_id='offline_only')
                try:
                    client.fetch({'action': 'query', 'format': 'json'}, name)
                    raise AssertionError('Expected simulated request failure')
                except RuntimeError:
                    pass
                count = request.call_count
            if name == 'lowercase_seconds':
                assert any(x >= 36 for x in waits) and count == 2
            elif name == 'http_date':
                assert any(35 <= x <= 50 for x in waits) and count == 2
            elif name == 'long_deferral':
                assert count == 1
            else:
                assert count == 1 and client.stop
            results.append(dict(case=name, fake_request_count=count, simulated_waits=waits))
    result = dict(status='passed', real_network_requests=0, raw_collection_unchanged=True,
                  source_sha256=hashlib.sha256((HERE / 'collect_metadata.py').read_bytes()).hexdigest(), cases=results)
    (HERE / 'retry_header_qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
