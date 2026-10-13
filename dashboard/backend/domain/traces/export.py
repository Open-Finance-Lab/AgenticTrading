"""Versioned, redacted trace exports with a fixed event-sequence boundary."""
import html
import json
import re
import shutil
from datetime import datetime, timezone
from tempfile import SpooledTemporaryFile

# Key matching is per word, not per substring: ``token`` must catch
# ``refresh_token`` and ``accessToken`` without blanking ``input_tokens`` or
# ``llm_max_output_tokens``, and ``session``/``browser`` are identities only
# as a whole key or with ``id`` -- ``open_session_excluded`` is configuration.
# Stems are sensitive anywhere in a key once separators and case are dropped
# (``x-api-key``, ``XAPIKey``, ``client_secret``).
_SENSITIVE_STEMS = (
    'apikey', 'authorization', 'bearer', 'cookie', 'credential', 'email',
    'passwd', 'password', 'privatekey', 'secret',
)
# Sensitive only as the end of a word or of two adjacent words, so
# ``session_id``, ``sessionId`` and ``sessionid`` all match.
_SENSITIVE_SUFFIXES = ('token', 'sessionid', 'browserid', 'userid', 'actorid')
_SENSITIVE_KEYS = frozenset({'session', 'browser'})
_CAMEL = re.compile(r'([a-z0-9])([A-Z])')
# The lookbehind starts a match only at the head of a run: without it a long
# run with no ``@`` is re-scanned from every offset, quadratic in its length,
# and agent text reaches this up to the store's 64 KiB payload cap.
_EMAIL = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}')
_BEARER = re.compile(r'(?i)\bBearer\s+[^\s"<>]+')
_KEY = re.compile(r'\b(?:sk-|sk_)[A-Za-z0-9_-]{12,}')
# Inline constructs an agent could use to plant a link, an image beacon or
# formatting in an admin's Markdown viewer. Line-start constructs (headings,
# lists, setext rules) are prevented by folding newlines, since every line this
# module writes starts with its own label. ``&``, ``<`` and ``>`` are left to
# ``html.escape``, whose entities must not be broken apart.
_MARKDOWN_PUNCTUATION = re.compile(r'([\\`*_\[\]()!~])')
_TERMINAL_EVENTS = {'run_completed': 'completed', 'run_failed': 'failed'}
_RUN_STATUS_LINE = {
    'completed': 'Run completed.',
    'failed': 'Run failed.',
    'running': 'Run is incomplete: it was still running at capture.',
}


def _key_words(key):
    return [word for word in re.split(r'[^a-z0-9]+', _CAMEL.sub(r'\1_\2', str(key)).lower()) if word]


def _is_sensitive_key(key):
    words = _key_words(key)
    compact = ''.join(words)
    if compact in _SENSITIVE_KEYS or any(stem in compact for stem in _SENSITIVE_STEMS):
        return True
    candidates = words + [first + second for first, second in zip(words, words[1:])]
    return any(word.endswith(_SENSITIVE_SUFFIXES) for word in candidates)


def redact(value):
    if isinstance(value, dict):
        return {key: '[REDACTED]' if _is_sensitive_key(key) else redact(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        return _KEY.sub('[REDACTED]', _BEARER.sub('[REDACTED]', _EMAIL.sub('[REDACTED]', value)))
    return value


def markdown_text(value):
    """Agent-supplied text as inert Markdown: no links, images, or headings.
    The exact text stays in the event's escaped ``<pre>`` block."""
    flat = ' '.join(str(value).splitlines())
    return _MARKDOWN_PUNCTUATION.sub(r'\\\1', html.escape(flat, quote=False))


def run_status(trace, events_status):
    """The run's state as of the export boundary.

    A terminal event inside the boundary is authoritative: the trace row is read
    before the boundary is fixed, and the store appends the terminal event before
    it updates the row, so a row can still say ``running`` for an export that
    holds ``run_completed``. A terminal row with no terminal event (a store that
    never wrote one) is believed as read."""
    if events_status:
        return events_status
    status = (trace or {}).get('status')
    return status if status in ('completed', 'failed') else 'running'


def build_export(store, trace, format):
    """Build completely before sending headers; spill larger files to disk."""
    captured_at = datetime.now(timezone.utc).isoformat()
    cutoff = store.event_high_watermark(trace['trace_id'])
    body = SpooledTemporaryFile(max_size=1024 * 1024, mode='w+b')
    file = SpooledTemporaryFile(max_size=1024 * 1024, mode='w+b')

    def write(target, text):
        target.write(text.encode('utf-8'))

    def dump(value):
        return json.dumps(value, ensure_ascii=False, indent=2)

    def block(value):
        # An HTML-safe code block even when a message contains backticks.
        return '\n<pre>\n' + html.escape(dump(value)) + '\n</pre>\n'

    try:
        cursor = 0
        first = True
        events_status = None
        while cursor < cutoff:
            page = store.list_events(trace['trace_id'], after_sequence=cursor, limit=100)
            rows = [e for e in page['items'] if cursor < e['sequence_no'] <= cutoff]
            if not rows:
                raise RuntimeError('Trace changed or event page missing during export')
            for event in rows:
                events_status = _TERMINAL_EVENTS.get(event.get('event_type'), events_status)
                clean = redact(event)
                if format == 'json':
                    write(body, ('' if first else ',\n') + dump(clean))
                else:
                    write(body, f'\n### Event {event["sequence_no"]}: {markdown_text(clean.get("event_type", "event"))}\n\n')
                    write(body, 'Recorded: ' + markdown_text(clean.get('occurred_at', 'Not recorded')) + '\n\n')
                    payload = clean.get('payload') or {}
                    if isinstance(payload, dict):
                        reasons = payload.get('reasoning_summaries') or [payload.get('reasoning_summary')]
                        if isinstance(reasons, list):
                            for reason in reasons:
                                if isinstance(reason, str) and reason:
                                    write(body, 'Reason: ' + markdown_text(reason) + '\n\n')
                        for label in ('error_code', 'message', 'fills', 'actions', 'rejected'):
                            detail = payload.get(label)
                            if not detail:
                                continue
                            name = label.replace('_', ' ').title()
                            if isinstance(detail, str):
                                write(body, f'{name}: {markdown_text(detail)}\n\n')
                            else:
                                # Structured values carry agent text too (an
                                # action's reasoning), so they go in a block.
                                write(body, f'{name}:\n{block(detail)}\n')
                    write(body, '<details><summary>Technical details</summary>\n' + block(clean) + '\n</details>\n')
                first = False
            cursor = rows[-1]['sequence_no']
        # The header is written after the walk so its status describes exactly
        # the events through ``cutoff``, not the row as it was read earlier.
        status = run_status(trace, events_status)
        exported_trace = redact({**trace, 'status': status})
        header = {'schema_version': 1, 'captured_at': captured_at,
                  'through_sequence': cutoff, 'run_status': status,
                  'incomplete_run': status == 'running',
                  'artifact_policy': 'References and metadata only; attachments are not bundled.',
                  'trace': exported_trace}
        if format == 'json':
            write(file, dump(header)[:-2] + ',\n  "events": [\n')
        else:
            write(file, '# Agent trace report\n\n')
            write(file, f'Captured at: {markdown_text(captured_at)}\n\nThrough event sequence: {cutoff}\n\n')
            write(file, _RUN_STATUS_LINE[status] + '\n\n')
            write(file, 'Artifact references only; attachments are not bundled.\n\n## Run overview\n' + block(exported_trace) + '\n## Timeline\n')
        body.seek(0)
        shutil.copyfileobj(body, file)
        if format == 'json':
            write(file, '\n]}\n')
        file.seek(0)
        return file
    except Exception:
        file.close()
        raise
    finally:
        body.close()
