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
        code: error.code,
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
        name: 'FastAPI detail object',
        response: () => errorResponse(409, {
            detail: {
                loc: ['body', 'name'],
                msg: 'Name is already in use.',
                type: 'conflict',
            },
        }),
        expectedMessage: 'Request failed. body.name: Name is already in use.',
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
    assert.equal(error.code, 'validation_error');
    assert.equal(error.correlationId, 'req_01J.safe:123');
    assert.equal(error.message, 'Request validation failed. body.name: Field required (Reference: req_01J.safe:123)');
    assert.deepEqual(error.details, [{ message: 'Field required', location: 'body.name', code: 'missing' }]);
});

test('unsafe response correlation header falls back to the canonical envelope ID', async () => {
    const error = await captureRequestError(() => errorResponse(
        409,
        {
            error: {
                code: 'conflict',
                message: 'Already exists.',
                correlation_id: 'safe-envelope-id',
            },
        },
        { correlationId: '<script>unsafe</script>' },
    ));

    assert.equal(error.code, 'conflict');
    assert.equal(error.correlationId, 'safe-envelope-id');
    assert.equal(error.message, 'Already exists. (Reference: safe-envelope-id)');
});

test('structured details are bounded, normalized, and never stringify object locations', async () => {
    const details = Array.from({ length: 25 }, (_, index) => ({
        location: index === 0 ? [{ private: 'object' }, 'field'] : ['body', index],
        message: index === 0
            ? '<b>Denied</b> Bearer super-secret-token'
            : `Problem ${index}`,
        code: `problem_${index}`,
    }));
    const error = await captureRequestError(() => errorResponse(422, {
        error: {
            code: 'validation_error',
            message: 'Request validation failed.',
            details,
        },
    }));

    assert.equal(error.details.length, 20);
    assert.deepEqual(error.details[0], {
        message: 'Denied Bearer [redacted]',
        location: 'field',
        code: 'problem_0',
    });
    assert.doesNotMatch(error.message, /<b>|super-secret-token|\[object Object\]/);
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

test('unterminated HTML-like response text is not exposed as an error message', async () => {
    const error = await captureRequestError(
        () => errorResponse(502, '<img src=x onerror=secret=leaked', {
            json: false,
            statusText: 'Bad Gateway',
        }),
    );

    assert.equal(error.message, 'Bad Gateway');
    assert.doesNotMatch(error.message, /<img|secret|leaked/i);
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

test('stream 401 refresh retries once and preserves the retried error contract', async () => {
    const responses = [
        errorResponse(401, { detail: 'expired' }),
        errorResponse(
            403,
            { error: { code: 'still_forbidden', message: 'Still forbidden.' } },
            { correlationId: 'stream-retry-403' },
        ),
    ];
    let unauthorizedCalls = 0;
    const client = clientFor(async () => responses.shift(), {
        async ensureAuthenticated() {},
        async applyAuth(value) { return value; },
        async onUnauthorized() { unauthorizedCalls += 1; return 'refreshed'; },
    });

    await assert.rejects(
        () => client.streamInvoke('hello').next(),
        (error) => {
            assert.ok(error instanceof ApiError);
            assert.equal(error.status, 403);
            assert.equal(error.code, 'still_forbidden');
            assert.equal(error.message, 'Still forbidden. (Reference: stream-retry-403)');
            return true;
        },
    );
    assert.equal(unauthorizedCalls, 1);
});

for (const mode of ['request', 'stream']) {
    test(`${mode} preserves the original 401 when auth recovery throws`, async () => {
        const client = clientFor(async () => errorResponse(
            401,
            { error: { code: 'authentication_required', message: 'Sign in again.' } },
            { correlationId: 'auth-refresh-failed' },
        ), {
            async ensureAuthenticated() {},
            async applyAuth(value) { return value; },
            async onUnauthorized() { throw new Error('refresh callback leaked a secret'); },
        });

        const operation = mode === 'stream'
            ? () => client.streamInvoke('hello').next()
            : () => client.request('/api/example');
        await assert.rejects(operation, (error) => {
            assert.ok(error instanceof ApiError);
            assert.equal(error.status, 401);
            assert.equal(error.code, 'authentication_required');
            assert.equal(error.correlationId, 'auth-refresh-failed');
            assert.equal(error.message, 'Sign in again. (Reference: auth-refresh-failed)');
            assert.doesNotMatch(error.message, /secret/i);
            return true;
        });
    });
}

for (const mode of ['request', 'stream']) {
    test(`${mode} preserves the original typed 401 when refreshed auth header setup throws`, async () => {
        const setupFailures = [
            new Error('credential provider leaked a refresh secret'),
            new ApiError({
                status: 503,
                body: { secret: 'credential-provider-internal' },
                code: 'provider_failure',
                message: 'Credential provider secret failure.',
            }),
        ];

        for (const setupFailure of setupFailures) {
            let applyAuthCalls = 0;
            let fetchCalls = 0;
            const client = clientFor(async () => {
                fetchCalls += 1;
                return errorResponse(
                    401,
                    { error: { code: 'authentication_required', message: 'Session expired.' } },
                    { correlationId: 'original-401-after-refresh' },
                );
            }, {
                async ensureAuthenticated() {},
                async applyAuth(value) {
                    applyAuthCalls += 1;
                    if (applyAuthCalls === 2) throw setupFailure;
                    return value;
                },
                async onUnauthorized() { return 'refreshed'; },
            });

            const operation = mode === 'stream'
                ? () => client.streamInvoke('hello').next()
                : () => client.request('/api/example');
            await assert.rejects(operation, (error) => {
                assert.ok(error instanceof ApiError);
                assert.equal(error.status, 401);
                assert.equal(error.code, 'authentication_required');
                assert.equal(error.correlationId, 'original-401-after-refresh');
                assert.equal(error.message, 'Session expired. (Reference: original-401-after-refresh)');
                assert.doesNotMatch(error.message, /credential|secret|provider_failure/i);
                return true;
            });
            assert.equal(applyAuthCalls, 2);
            assert.equal(fetchCalls, 1);
        }
    });
}

for (const mode of ['request', 'stream']) {
    test(`${mode} preserves the original typed 401 when starting the refreshed retry throws`, async () => {
        let fetchCalls = 0;
        const client = clientFor(async () => {
            fetchCalls += 1;
            if (fetchCalls === 2) {
                throw new TypeError('retry transport exposed internal connection data');
            }
            return errorResponse(
                401,
                { error: { code: 'authentication_required', message: 'Session expired.' } },
                { correlationId: 'original-401-before-retry' },
            );
        }, {
            async ensureAuthenticated() {},
            async applyAuth(value) { return value; },
            async onUnauthorized() { return 'refreshed'; },
        });

        const operation = mode === 'stream'
            ? () => client.streamInvoke('hello').next()
            : () => client.request('/api/example');
        await assert.rejects(operation, (error) => {
            assert.ok(error instanceof ApiError);
            assert.equal(error.status, 401);
            assert.equal(error.code, 'authentication_required');
            assert.equal(error.correlationId, 'original-401-before-retry');
            assert.doesNotMatch(error.message, /transport|connection/i);
            return true;
        });
        assert.equal(fetchCalls, 2);
    });
}

test('stopping a stream during 401 refresh prevents the retry from resurrecting it', async () => {
    let finishRefresh;
    let markRefreshStarted;
    const refreshStarted = new Promise((resolve) => { markRefreshStarted = resolve; });
    let fetchCalls = 0;
    const client = clientFor(async () => {
        fetchCalls += 1;
        return errorResponse(401, { detail: 'expired' });
    }, {
        async ensureAuthenticated() {},
        async applyAuth(value) { return value; },
        async onUnauthorized() {
            markRefreshStarted();
            return new Promise((resolve) => { finishRefresh = resolve; });
        },
    });

    const pending = client.streamInvoke('hello').next();
    await refreshStarted;
    const controller = client.getStreamAbortController();
    assert.ok(controller);
    controller.abort();
    finishRefresh('refreshed');

    await assert.rejects(pending, (error) => error && error.name === 'AbortError');
    assert.equal(fetchCalls, 1);
    assert.equal(client.getStreamAbortController(), null);
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

for (const mode of ['request', 'stream']) {
    test(`${mode} preserves AbortError raised while consuming a 401 body`, async () => {
        const abort = new Error('cancelled while reading the response');
        abort.name = 'AbortError';
        let unauthorizedCalls = 0;
        const client = clientFor(async () => ({
            ok: false,
            status: 401,
            statusText: 'Unauthorized',
            headers: headers(),
            async text() { throw abort; },
        }), {
            async ensureAuthenticated() {},
            async applyAuth(value) { return value; },
            async onUnauthorized() { unauthorizedCalls += 1; return 'refreshed'; },
        });

        const operation = mode === 'stream'
            ? () => client.streamInvoke('hello').next()
            : () => client.request('/api/example');
        await assert.rejects(
            operation,
            (error) => error === abort && !(error instanceof ApiError),
        );
        assert.equal(unauthorizedCalls, 0);
    });
}

test('request cancellation after 401 body parsing skips auth recovery', async () => {
    const controller = new AbortController();
    let unauthorizedCalls = 0;
    const client = clientFor(async () => ({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        headers: headers(),
        async text() {
            controller.abort();
            return JSON.stringify({ detail: 'expired' });
        },
    }), {
        async ensureAuthenticated() {},
        async applyAuth(value) { return value; },
        async onUnauthorized() { unauthorizedCalls += 1; return 'refreshed'; },
    });

    await assert.rejects(
        () => client.request('/api/example', { signal: controller.signal }),
        (error) => error && error.name === 'AbortError' && !(error instanceof ApiError),
    );
    assert.equal(unauthorizedCalls, 0);
});

test('stream cancellation after 401 body parsing skips auth recovery', async () => {
    let client;
    let unauthorizedCalls = 0;
    client = clientFor(async () => ({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        headers: headers(),
        async text() {
            client.getStreamAbortController()?.abort();
            return JSON.stringify({ detail: 'expired' });
        },
    }), {
        async ensureAuthenticated() {},
        async applyAuth(value) { return value; },
        async onUnauthorized() { unauthorizedCalls += 1; return 'refreshed'; },
    });

    await assert.rejects(
        () => client.streamInvoke('hello').next(),
        (error) => error && error.name === 'AbortError' && !(error instanceof ApiError),
    );
    assert.equal(unauthorizedCalls, 0);
});
