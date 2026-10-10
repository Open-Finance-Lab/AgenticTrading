"""Versioned, redacted trace exports with a fixed event-sequence boundary."""
import html
import json
import re
from datetime import datetime, timezone
from tempfile import SpooledTemporaryFile

_PRIVATE = re.compile(r'api[_-]?key|authorization|cookie|password|secret|token|email|session|browser|user_id|actor_id', re.I)
_EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}')
_BEARER = re.compile(r'(?i)\bBearer\s+[^\s"<>]+')
_KEY = re.compile(r'\b(?:sk-|sk_)[A-Za-z0-9_-]{12,}')


def redact(value):
    if isinstance(value, dict):
        return {key: '[REDACTED]' if _PRIVATE.search(str(key)) else redact(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        return _KEY.sub('[REDACTED]', _BEARER.sub('[REDACTED]', _EMAIL.sub('[REDACTED]', value)))
    return value


def build_export(store, trace, format):
    """Build completely before sending headers; spill larger files to disk."""
    captured_at = datetime.now(timezone.utc).isoformat()
    cutoff = store.event_high_watermark(trace['trace_id'])
    header = {'schema_version': 1, 'captured_at': captured_at,
              'through_sequence': cutoff, 'incomplete_run': trace.get('status') != 'completed',
              'artifact_policy': 'References and metadata only; attachments are not bundled.',
              'trace': redact(trace)}
    file = SpooledTemporaryFile(max_size=1024 * 1024, mode='w+b')
    def write(text):
        file.write(text.encode('utf-8'))
    def dump(value):
        return json.dumps(value, ensure_ascii=False, indent=2)
    def block(value):
        # An HTML-safe code block even when a message contains backticks.
        return '\n<pre>\n' + html.escape(dump(value)) + '\n</pre>\n'
    try:
        if format == 'json':
            write(dump(header)[:-2] + ',\n  "events": [\n')
        else:
            write('# Agent trace report\n\n')
            write(f'Captured at: {captured_at}\n\nThrough event sequence: {cutoff}\n\n')
            write('Run is incomplete.\n\n' if header['incomplete_run'] else 'Run completed.\n\n')
            write('Artifact references only; attachments are not bundled.\n\n## Run overview\n' + block(header['trace']) + '\n## Timeline\n')
        cursor = 0
        first = True
        while cursor < cutoff:
            page = store.list_events(trace['trace_id'], after_sequence=cursor, limit=100)
            rows = [e for e in page['items'] if cursor < e['sequence_no'] <= cutoff]
            if not rows:
                raise RuntimeError('Trace changed or event page missing during export')
            for event in rows:
                clean = redact(event)
                if format == 'json':
                    write(('' if first else ',\n') + dump(clean))
                else:
                    write(f'\n### Event {event["sequence_no"]}: {html.escape(str(clean.get("event_type", "event")))}\n\n')
                    write('Recorded: ' + html.escape(str(clean.get('occurred_at', 'Not recorded'))) + '\n\n')
                    payload = clean.get('payload') or {}
                    if isinstance(payload, dict):
                        reasons = payload.get('reasoning_summaries') or [payload.get('reasoning_summary')]
                        if isinstance(reasons, list):
                            for reason in reasons:
                                if isinstance(reason, str) and reason:
                                    write('Reason: ' + html.escape(reason) + '\n\n')
                        for label in ('error_code', 'message', 'fills', 'actions', 'rejected'):
                            if payload.get(label):
                                write(html.escape(label.replace('_', ' ').title()) + ': ' + html.escape(dump(payload[label])) + '\n\n')
                    write('<details><summary>Technical details</summary>\n' + block(clean) + '\n</details>\n')
                first = False
            cursor = rows[-1]['sequence_no']
        if format == 'json':
            write('\n]}\n')
        file.seek(0)
        return file
    except Exception:
        file.close()
        raise
