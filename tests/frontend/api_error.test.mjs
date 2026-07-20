import test from 'node:test';
import assert from 'node:assert/strict';

import {
    ApiError,
    createApiClient,
} from '../../kestrel_sovereign/static/js/api_client.mjs';

function storage() {
    return { getItem() { return null; }, setItem() {}, removeItem() {} };
}

function headers(values = {}) {
    const entries = new Map(Object.entries(values).map(([key, value]) => [key.toLowerCase(), value]));
    return { get(name) { return entries.get(String(name).toLowerCase()) ?? null; } };
}

function errorResponse(status, body, {
    correlationId = null,
    statusText = `HTTP ${status}`,
    json = true,
} = {}) {
    const text = body == null ? '' : (json ? JSON.stringify(body) : String(body));
    return {
        ok: false,
        status,
        statusText,
        headers: headers(correlationId ? { 'X-Correlation-ID': correlationId } : {}),
        async text() { return text; },
    };
}

function clientFor(fetchFn, authProvider = null) {
    return createApiClient({
        fetchFn,
        sessionStorage: storage(),
        location: { href: '/', search: '' },
        logger: { log() {}, warn() {}, error() {} },
        authProvider: authProvider || {
            async ensureAuthenticated() {},
            async applyAuth(value) { return value; },
            async onUnauthorized() { return 'failed'; },
        },
    });
}

async function captureRequestError(makeResponse, mode = 'request') {
    const client = clientFor(async () => makeResponse());
    try {
        if (mode === 'stream') {
            await client.streamInvoke('hello').next();
        } else if (mode === 'host') {
            await client.requestHost('/api/host/example');
        } else {
            await client.request('/api/example');
        }
        assert.fail('request should reject');
    } catch (error) {
        return error;
    }
}

function comparable(error) {
    return {
        name: error.name,
        status: error.status,
        body: error.body,
        message: error.message,
        details: error.details,
        correlationId: error.correlationId,
    };
}

const RESPONSE_CASES = [
    {
        name: 'JSON detail string',
        response: () => errorResponse(
            400,
            { detail: 'Input not provided.' },
            { correlationId: 'same-stream-request-id' },
        ),
        expectedMessage: 'Input not provided. (Reference: same-stream-request-id)',
    },
    {
        name: 'FastAPI validation array',
        response: () => errorResponse(422, {
            detail: [
                { loc: ['body', 'input'], msg: 'Field required', type: 'missing', input: 'secret-value' },
                { loc: ['body', 'count'], msg: 'Must be positive', type: 'greater_than', ctx: { gt: 0 } },
            ],
        }),
        expectedMessage: 'Request validation failed. body.input: Field required; body.count: Must be positive',
    },
    {
        name: '{error} compatibility body',
        response: () => errorResponse(503, { error: 'GitHub integration is not configured.' }),
        expectedMessage: 'GitHub integration is not configured.',
    },
    {
        name: 'non-JSON text body',
        response: () => errorResponse(502, 'Upstream is temporarily unavailable.', { json: false }),
        expectedMessage: 'Upstream is temporarily unavailable.',
    },
    {
        name: 'empty body',
        response: () => errorResponse(500, null, { statusText: 'Internal Server Error' }),
        expectedMessage: 'Internal Server Error',
    },
];

for (const scenario of RESPONSE_CASES) {
    test(`stream and non-stream produce equivalent ApiError fields: ${scenario.name}`, async () => {
        const requestError = await captureRequestError(scenario.response);
        const streamError = await captureRequestError(scenario.response, 'stream');

        assert.ok(requestError instanceof ApiError);
        assert.ok(streamError instanceof ApiError);
        assert.deepEqual(comparable(streamError), comparable(requestError));
        assert.equal(requestError.message, scenario.expectedMessage);
        assert.doesNotMatch(requestError.message, /secret-value|\[object Object\]/);
    });
}

test('canonical envelope retains body, structured details, and support-safe correlation ID', async () => {
    const body = {
        error: {
            code: 'validation_error',
            message: 'Request validation failed.',
            details: [{ location: ['body', 'name'], message: 'Field required', code: 'missing' }],
        },
        detail: [{ location: ['body', 'name'], message: 'Field required', code: 'missing' }],
    };
    const error = await captureRequestError(
        () => errorResponse(422, body, { correlationId: 'req_01J.safe:123' }),
    );

    assert.equal(error.status, 422);
    assert.deepEqual(error.body, body);
    assert.equal(error.correlationId, 'req_01J.safe:123');
    assert.equal(error.message, 'Request validation failed. body.name: Field required (Reference: req_01J.safe:123)');
    assert.deepEqual(error.details, [{ message: 'Field required', location: 'body.name', code: 'missing' }]);
});

test('host requests use the same parser and never expose raw HTML or credential assignments', async () => {
    const error = await captureRequestError(
        () => errorResponse(
            502,
            '<html><body>proxy secret=super-secret-value</body></html>',
            { json: false, statusText: 'Bad Gateway' },
        ),
        'host',
    );

    assert.ok(error instanceof ApiError);
    assert.equal(error.body, '<html><body>proxy secret=super-secret-value</body></html>');
    assert.equal(error.message, 'Bad Gateway');
    assert.doesNotMatch(error.message, /html|super-secret-value/i);
});

test('401 refresh retries once and a failed retried response is still an ApiError', async () => {
    const responses = [
        errorResponse(401, { detail: 'expired' }),
        errorResponse(403, { detail: 'Still forbidden.' }, { correlationId: 'retry-403' }),
    ];
    let unauthorizedCalls = 0;
    const client = clientFor(async () => responses.shift(), {
        async ensureAuthenticated() {},
        async applyAuth(value) { return value; },
        async onUnauthorized() { unauthorizedCalls += 1; return 'refreshed'; },
    });

    await assert.rejects(
        () => client.request('/api/example'),
        (error) => {
            assert.ok(error instanceof ApiError);
            assert.equal(error.status, 403);
            assert.equal(error.message, 'Still forbidden. (Reference: retry-403)');
            return true;
        },
    );
    assert.equal(unauthorizedCalls, 1);
});

test('stream setup preserves AbortError instead of wrapping it as ApiError', async () => {
    const abort = new Error('stopped');
    abort.name = 'AbortError';
    const client = clientFor(async () => { throw abort; });

    await assert.rejects(
        () => client.streamInvoke('hello').next(),
        (error) => error === abort && !(error instanceof ApiError),
    );
});
